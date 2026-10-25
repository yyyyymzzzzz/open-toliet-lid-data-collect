# robot-5 实用采集工作台：分段代理约定

工作目录 W=`/Users/ymz/Documents/ChatGPT/open toilet lid/new_robot_r5/dashboard`。先读 FACTS.md、theme.css、dp.js。本任务显式使用用户的 codebase-visualization skill，授权分段代理。用户要求的是可操作工作台，优先功能，中文，面向现场操作者。

## 文件所有权

每个代理只写分配的一个 `sections/NN-name.html`，以及自己在 `qa/<name>/` 的截图。不要修改共享文件或后端。协调者写 shell、hero、theme补充、app.js、后端和服务。禁止机器人操作、远端部署、Git修改、读取凭据、启动服务。可以运行只生成自己preview的build和截图。

## 截图（必做）

技能的setup脚本写死Linux x86，本机为Mac ARM，因此已在现有隔离环境安装Playwright/Pillow，使用已安装Chrome。shot.py保留DP_STATIC行为并增加可配置浏览器路径/HTTP URL。

```
cd "$W"
python3 build.py --only NN-name
DPVIZ_ROOT="$W/qa" DPVIZ_CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' /private/tmp/pi05-r5-verify/bin/python shot.py preview/NN-name.html --section name --width 1440 --out qa/name/1440.png
# 同上--width 1000，再看图。若browser launch被沙箱限制，可请求执行该本地截图的权限，或报告协调者统一检查。
```

## 界面与片段

一个 `<section class="sec" id="name" data-nav="中文">`，内部 `.wrap`；后接style所有selector以该id开头，以及IIFE脚本。主题是技能theme.css。共享增加 `.btn`, `.btn.primary`, `.btn.danger`, `.field`, `.empty`, `.status-pill`, `.metric`, `.toolbar`, `.sr-only`。使用语义label/button和aria。键盘可操作，错误文字能看懂。

SVG手工绘图，无CDN/外部库；色彩沿theme。将技能大量白边收紧为实用面板，协调者会把hero压缩和导航处理为三页tabs；每个section可单独使用。caption说明数字含义，行内状态不假装成功。

## 共享JS接口（协调者实现）

`window.R5` 在片段之前提供：
- `R5.on(fn)`：订阅状态；fn(state) 每次刷新触发，也立即发当前值。
- `R5.state`：`{connected, status, episodes:[], exports:[], evaluations:[]}`。
- `R5.api('/api/...')`：GET返回Promise JSON，失败抛Error。
- `R5.post('/api/...', payload)`：POST同上。JS传普通对象。
- `R5.refresh()`：刷新状态并通知。
- `R5.toast(text, isError=false)`：通知。
- `R5.escape(value)`：安全HTML转义；所有从API取得的字符串必须textContent或escape，禁止裸innerHTML注入。
- `R5.fmt(value,digits=1)`：数字格式，null/undefined=>`—`。
- `R5.chart(element, series, {unit,height=190})`：折线SVG；series=`[{name,color,points:[[x,y],...]}]`。坐标轴自动范围，x秒/step由调用处标题说明。
- `R5.selectTab('capture'|'library'|'evaluation')`。

无API时connected=false，所有写按钮禁用，空态说明“等待机器人连接”。不要自己生成fake数据。

## API契约

GET `/api/status` 返回：
```
{now, mode:'IDLE'|'RECORDING'|'REVIEW'|'ERROR',
 recorder:{id:null|string,frames:0,elapsed_s:0,error:null|string,diagnostic:false},
 observation:{ready:false,reason:'...',state:null|[17],cameras:{cam_bottom:{available,age_s,hz},cam_wrist:{...}}},
 disks:{system:{free_bytes,total_bytes},data:{free_bytes,total_bytes}},
 summary:{total,accepted,diagnostic,frames},
 contract:{fps:30,state_dim:17,action_dim:7,horizon:16}, training:{connected:false}}
```
POST `/api/record/start` `{diagnostic:boolean}`；POST `/api/record/stop` `{}`。采集仅被动记录，stop不是急停机器人。
GET `/api/live/frame?camera=cam_bottom&t=<time>` 返回JPEG，另一路cam_wrist。建议图像每500ms刷新；connected=false时不请求。错误/过期由status标注。

GET `/api/episodes` => `{episodes:[{id,created_at,frames,actual_source_hz,max_gap_s,status:'ACCEPTED'|'PENDING'|'REJECTED'|'DIAGNOSTIC'|'FAILED'|'RECORDING',diagnostic,accepted,quality:{status,errors:[]},task}]}`。
GET `/api/episodes/<id>` => `{episode:<summary>,frame_count,series:[{frame_index,t,joints:[6],xyz:[3],dt_ms}],quality}`；series最多1000点，frame_count是原始总帧数。
GET `/api/episodes/<id>/frame?camera=cam_bottom&index=<int>` =>原始JPEG缩放后的图片（另一路wrist）。可拖动时间线；不需要视频播放控件。
POST `/api/episodes/<id>/review` `{decision:'accept'|'reject',note:string}`。diagnostic或FAIL不能accept；没有删除API，reject只改接受标记。

GET `/api/exports` => `{exports:[{id,name,status:'RUNNING'|'DONE'|'FAILED',frames,episodes,error,path,created_at}]}`。
POST `/api/exports` `{name:'<可选安全名称>'}` 只导出accepted+PASS+非diagnostic；异步后台任务。录制/导出忙时会拒绝。

GET `/api/evaluations` => `{evaluations:[{id,name,checkpoint,dataset,created_at,training:[{step,loss,validation_loss?}],evaluation:{successes?,trials?,mean_joint_mae_deg?,mean_latency_ms?,per_joint_mae_deg?:[6]},notes,source:'user_import'}]}`。
POST `/api/evaluations/import` `{run:<同上但id/created_at/source由后端生成>}`。支持选择JSON文件；必须校验实际数字。没有GPU连接，不提供假启动训练按钮。可以给可下载的空schema模板，示例数值用null或空数组，不能把示例展示为真实结果。

## 分工

| file | id | 主题 |
|---|---|---|
| 10-capture.html | capture | 双路实时画面、录制控制、状态/关节读数、帧率/新鲜度、工作流步骤 |
| 20-library.html | library | 样本筛选列表、双路逐帧回看+关节/时间间隙图、接受/拒绝、导出任务 |
| 30-evaluation.html | evaluation | 训练loss/验证loss曲线、评测指标比较、真实JSON导入、无GPU/无结果的诚实空态 |

最后报告：视觉布局、使用的事实、是否截图检查、任何后端契约建议。不要修改其它代理文件。
