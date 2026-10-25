"""Read the deployed URDF and its packaged STL visuals; no robot connection."""
from pathlib import Path
import hashlib
import math
import struct
import xml.etree.ElementTree as ET
import numpy as np


def numbers(text, default):
    out=[float(v) for v in text.split()] if text else list(default)
    if len(out)!=len(default) or not all(math.isfinite(v) for v in out):raise ValueError('Invalid URDF vector')
    return out


def origin(node):
    o=node.find('origin')
    return dict(xyz=numbers(o.get('xyz') if o is not None else None,[0,0,0]),
                rpy=numbers(o.get('rpy') if o is not None else None,[0,0,0]))


def stl_vertices(path):
    raw=path.read_bytes()
    if len(raw)>=84 and 84+50*struct.unpack_from('<I',raw,80)[0]==len(raw):
        dtype=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attr','<u2')])
        vertices=np.frombuffer(raw,offset=84,dtype=dtype)['vertices'].reshape(-1,3)
    else:
        vertices=np.array([[float(v) for v in line.split()[1:]] for line in raw.decode('ascii').splitlines() if line.strip().startswith('vertex ')])
    if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)%3 or not np.isfinite(vertices).all():raise ValueError('Invalid STL geometry')
    return np.round(vertices.astype(np.float64),6).reshape(-1).tolist()


def load_model():
    import yaml
    from ament_index_python.packages import get_package_share_directory
    config=Path(get_package_share_directory('arm_control'))/'config/robot_config.yaml'
    arm=yaml.safe_load(config.read_text())['arm']
    path=Path(get_package_share_directory(arm['description_package']))/arm['urdf_file']
    return parse_model(path,get_package_share_directory)


def parse_model(path,package_lookup):
    raw=Path(path).read_bytes();root=ET.fromstring(raw)
    joints=[]
    for j in root.findall('joint'):
        axis=j.find('axis');lim=j.find('limit')
        joints.append(dict(name=j.get('name'),type=j.get('type'),parent=j.find('parent').get('link'),child=j.find('child').get('link'),
                           **origin(j),axis=numbers(axis.get('xyz') if axis is not None else None,[1,0,0]),
                           lower=float(lim.get('lower','0')) if lim is not None else 0,
                           upper=float(lim.get('upper','0')) if lim is not None else 0))
    reachable={'base_link_arm'}
    for _ in joints:
        reachable.update(j['child'] for j in joints if j['parent'] in reachable)
    joints=[j for j in joints if j['parent'] in reachable]
    materials={m.get('name'):numbers(m.find('color').get('rgba'),[.75,.78,.85,1]) for m in root.findall('material') if m.find('color') is not None}
    links=[];cache={}
    for link in root.findall('link'):
        if link.get('name') not in reachable:continue
        visuals=[]
        for v in link.findall('visual'):
            mesh=v.find('geometry/mesh')
            if mesh is None:raise ValueError('Unsupported non-mesh visual in deployed URDF')
            uri=mesh.get('filename','')
            if not uri.startswith('package://'):raise ValueError('Expected packaged STL')
            package,relative=uri[len('package://'):].split('/',1)
            package_root=Path(package_lookup(package)).resolve()
            mesh_path=(package_root/relative).resolve()
            if not mesh_path.is_relative_to(package_root):raise ValueError('Mesh escapes package')
            if mesh_path not in cache:cache[mesh_path]=stl_vertices(mesh_path)
            material=v.find('material');color=v.find('material/color')
            rgba=numbers(color.get('rgba'),[.75,.78,.85,1]) if color is not None else materials.get(material.get('name') if material is not None else '',[.75,.78,.85,1])
            visuals.append(dict(**origin(v),scale=numbers(mesh.get('scale'),[1,1,1]),color=rgba,vertices=cache[mesh_path],mesh=uri))
        links.append(dict(name=link.get('name'),visuals=visuals))
    return dict(root='base_link_arm',source=str(path),sha256=hashlib.sha256(raw).hexdigest(),links=links,joints=joints,
                gripper_display='URDF upper limits as configuration illustration; aperture not measured')


if __name__=='__main__':
    import gzip,json,os,sys
    try:os.nice(10)
    except (AttributeError,OSError):pass
    sys.stdout.buffer.write(gzip.compress(json.dumps(load_model(),separators=(',',':'),allow_nan=False).encode(),compresslevel=3))
