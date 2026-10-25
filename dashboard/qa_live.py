"""Visual/interaction QA. Optional recording is passive and permanently diagnostic."""
import argparse,json,os,time
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18765/');p.add_argument('--out',type=Path,default=Path('qa/live'));p.add_argument('--exercise-diagnostic',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    errors=[];recorded=None;report={}
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=os.environ.get('DPVIZ_CHROME') or '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args=['--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1440,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.add_init_script('window.DP_STATIC=true;')
        try:
            page.goto(a.url,wait_until='domcontentloaded');page.add_style_tag(content='.reveal{opacity:1!important;transform:none!important}.topnav{position:static!important}');page.wait_for_function('window.R5?.state.connected && Array.isArray(R5.state.status?.observation.state)',timeout=30000)
            page.wait_for_function('[...document.querySelectorAll("#capture img")].every(x=>x.complete && x.naturalWidth>0)',timeout=15000)
            if a.exercise_diagnostic:
                status=page.evaluate('R5.state.status')
                if status['mode']=='RECORDING':raise RuntimeError('A recording already exists; QA will not interrupt it')
                page.locator('#capture-diagnostic').check();page.locator('#capture-start').click()
                page.wait_for_function('R5.state.status?.mode==="RECORDING"',timeout=10000)
                recorded=page.evaluate('R5.state.status.recorder.id');time.sleep(4)
                page.locator('#capture-stop').click();page.wait_for_function('R5.state.status?.mode!=="RECORDING"',timeout=15000)
                result=page.evaluate('(id)=>R5.api("/api/episodes/"+id)',recorded)
                assert result['quality']['diagnostic'] and not result['quality']['accepted']
                assert result['quality']['status']=='PASS',result['quality']
                report['passive_diagnostic']=result['quality'];report['episode_id']=recorded
            for width in [1440,1000]:
                page.set_viewport_size({'width':width,'height':1000})
                for section in ['motion','capture','library','evaluation']:
                    page.evaluate('(id)=>R5.selectTab(id)',section);page.wait_for_timeout(900)
                    if section=='library':
                        buttons=page.locator('#lib-list .lib-item')
                        if buttons.count():
                            buttons.first.click();page.wait_for_function('!document.querySelector("#lib-scrubber").disabled')
                            page.locator('#lib-scrubber').fill('10');page.locator('#lib-scrubber').dispatch_event('input');page.wait_for_timeout(400)
                            assert page.locator('#lib-accept').is_disabled()
                    overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
                    assert not overflow,(section,width,'horizontal overflow')
                    page.locator('#'+section).screenshot(path=str(a.out/f'{section}_{width}.png'))
                    if section=='capture':
                        page.evaluate('scrollTo(0,0)');page.screenshot(path=str(a.out/f'overview_{width}.png'))
            report.update(javascript_errors=errors,motion_buttons_clicked=0,gripper_confirmations_submitted=0,training_runs_created=0)
            assert not errors,errors
            (a.out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
            print(json.dumps(report,indent=2,ensure_ascii=False))
        finally:
            if recorded:
                try:
                    page.evaluate('async(id)=>{let s=await R5.api("/api/status");if(s.mode==="RECORDING"&&s.recorder.id===id)await R5.post("/api/record/stop",{});}',recorded)
                except Exception:pass
            browser.close()


if __name__=='__main__':main()
