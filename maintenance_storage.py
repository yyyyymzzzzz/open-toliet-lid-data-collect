"""One-time, explicit robot-5 storage maintenance. No robot control operations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import grp
import shutil
import stat
import subprocess
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.digest()


def relocate(source, destination):
    if source.is_symlink():
        if source.resolve()!=destination.resolve():raise RuntimeError('Unexpected existing link: '+str(source))
        return
    if destination.exists():raise RuntimeError('Destination already exists: '+str(destination))
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source.exists():shutil.move(str(source),str(destination))
    else:destination.mkdir()
    source.parent.mkdir(parents=True,exist_ok=True)
    source.symlink_to(destination,target_is_directory=True)


def user_storage():
    if os.getuid()!=pwd.getpwnam('xzrobot').pw_uid:raise RuntimeError('Run user mode as xzrobot')
    base=Path('/mnt/Data/xzrobot');base.mkdir(exist_ok=True);base.chmod(0o700)
    relocate(Path('/home/xzrobot/workspace'),Path('/mnt/Data/workspace'))
    # The old parent /home/xzrobot was 0750; retain that privacy boundary on the data disk.
    Path('/mnt/Data/workspace').chmod(0o750)
    for name in ('uv','pip','huggingface','torch'):
        relocate(Path('/home/xzrobot/.cache')/name,base/'cache'/name)
    for name in ('tmp','env','migration_backups','uv-python'):
        (base/name).mkdir(exist_ok=True);(base/name).chmod(0o700)
    env=base/'env/storage.sh'
    env.write_text('''# robot-5: user projects and large caches live on the NVMe data disk.
export XDG_CACHE_HOME=/mnt/Data/xzrobot/cache
export UV_CACHE_DIR=/mnt/Data/xzrobot/cache/uv
export PIP_CACHE_DIR=/mnt/Data/xzrobot/cache/pip
export HF_HOME=/mnt/Data/xzrobot/cache/huggingface
export TORCH_HOME=/mnt/Data/xzrobot/cache/torch
export UV_PYTHON_INSTALL_DIR=/mnt/Data/xzrobot/uv-python
export TMPDIR=/mnt/Data/xzrobot/tmp
''')
    marker='# robot-5 data-disk storage defaults'
    block=marker+'\n. /mnt/Data/xzrobot/env/storage.sh\n'
    for name in ('.bashrc','.profile'):
        p=Path('/home/xzrobot')/name
        content=p.read_text() if p.exists() else ''
        if marker not in content:
            backup=base/'migration_backups'/(name+'.before-storage')
            if p.exists() and not backup.exists():shutil.copy2(p,backup);backup.chmod(0o600)
            # SSH's noninteractive bash reads .bashrc but its early return skips appended exports.
            p.write_text(block+'\n'+content if name=='.bashrc' else content+'\n'+block)
    for directory,filename,text in [
        ('uv','uv.toml','cache-dir = "/mnt/Data/xzrobot/cache/uv"\n'),
        ('pip','pip.conf','[global]\ncache-dir = /mnt/Data/xzrobot/cache/pip\n')]:
        folder=Path('/home/xzrobot/.config')/directory;folder.mkdir(exist_ok=True)
        target=folder/filename
        if target.exists() and target.read_text()!=text:raise RuntimeError('Existing package-manager configuration needs merge: '+str(target))
        target.write_text(text)
    print(json.dumps({'workspace':'/mnt/Data/workspace','compatibility_link':'/home/xzrobot/workspace',
                      'cache_root':str(base/'cache'),'shell_defaults':str(env)},indent=2))


def root_storage():
    if os.geteuid()!=0:raise RuntimeError('Root mode needs sudo')
    root=Path('/mnt/Data/system-logs')
    root.mkdir(exist_ok=True);os.chown(root,0,grp.getgrnam('adm').gr_gid);root.chmod(0o755)
    archive=root/'archive/20261024-cleanup';archive.mkdir(parents=True,exist_ok=True);archive.chmod(0o700)
    uid=pwd.getpwnam('syslog').pw_uid;gid=grp.getgrnam('adm').gr_gid
    rotated=[]
    for name in ('syslog','kern.log'):
        source=Path('/var/log')/name;dest=root/name
        if source.is_symlink():
            if source.resolve()!=dest:raise RuntimeError('Unexpected log link')
            continue
        if dest.exists():raise RuntimeError('Log destination already exists')
        original=source.with_name(name+'.before-data-migration')
        if original.exists():raise RuntimeError('Previous rotation requires reconciliation')
        if not stat.S_ISREG(source.lstat().st_mode):raise RuntimeError('Expected regular log')
        dest.touch(mode=0o640);os.chown(dest,uid,gid)
        source.rename(original);source.symlink_to(dest)
        rotated.append(original)
    if rotated:
        subprocess.run(['/usr/lib/rsyslog/rsyslog-rotate'],check=True)
        pid=int(subprocess.check_output(['systemctl','show','-p','MainPID','--value','rsyslog'],text=True))
        if pid<=0:raise RuntimeError('No rsyslog PID')
        for _ in range(100):
            targets=[]
            for fd in Path(f'/proc/{pid}/fd').iterdir():
                try:targets.append(os.readlink(fd))
                except FileNotFoundError:pass
            if not any(str(p) in targets for p in rotated):break
            time.sleep(.1)
        else:raise RuntimeError('rsyslog did not release old logs; original files retained')
        for old in rotated:
            dst=archive/old.name
            if dst.exists():raise RuntimeError('Archive already exists')
            size=old.stat().st_size
            shutil.copy2(old,dst)
            with dst.open('rb') as f:os.fsync(f.fileno())
            if size!=old.stat().st_size or digest(old)!=digest(dst):raise RuntimeError('Archive verification failed; original retained')
            old.unlink()
            print(json.dumps({'archived':str(dst),'bytes_freed':size}),flush=True)
    dropin=Path('/etc/systemd/system/rsyslog.service.d/60-data-disk.conf')
    dropin.parent.mkdir(parents=True,exist_ok=True)
    config='[Unit]\nRequiresMountsFor=/mnt/Data/system-logs\n'
    if dropin.exists() and dropin.read_text()!=config:raise RuntimeError('Unexpected existing rsyslog drop-in')
    dropin.write_text(config)
    journal=Path('/etc/systemd/journald.conf.d/60-local-disk-budget.conf')
    journal.parent.mkdir(parents=True,exist_ok=True)
    budget='[Journal]\nSystemMaxUse=200M\nSystemKeepFree=1G\nRuntimeMaxUse=100M\n'
    if journal.exists() and journal.read_text()!=budget:raise RuntimeError('Unexpected existing journal budget')
    journal.write_text(budget)
    subprocess.run(['systemctl','daemon-reload'],check=True)
    subprocess.run(['systemctl','restart','systemd-journald'],check=True)
    subprocess.run(['systemctl','is-active','rsyslog','systemd-journald'],check=True)
    print(json.dumps({'active_logs':str(root),'journal_budget':'200M','robot_services_restarted':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['user','root']);args=p.parse_args()
    (user_storage if args.mode=='user' else root_storage)()
