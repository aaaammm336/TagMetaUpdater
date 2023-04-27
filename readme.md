TagMetaUpdater
=
<br>
<img src="img\example.PNG"><br>
이미지의 태그를 인식해서 XMP meta데이터를 수정하는 프로그램 입니다<br>
<br>
<br>
첫 실행시 https://huggingface.co/SmilingWolf 에서 모델을 받아옵니다<br>
모델은 캐시에 저장되기 때문에 webui에 있는 wd14 tagger를 사용했다면 이 과정은 생략됩니다<br>
(캐시 주소 : C:\Users\사용자명\.cache\huggingface\hub)<br>
<br>
<br>
각 모델의 차이점은 https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/master/docs/model-comparison.md 을 참고해 주세요<br>
<br>
<br>
이미지를 인식하는 코드와 모델은 https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags 에서 가져왔으며<br>
그외의 코드는 chatGPT가 작성하였습니다<br>
<br>
<br>
<br>
<h2>사용방법</h2>
<img src="img\system_local.PNG"><br>
시스템 로컬을 UTF-8로 변경해야 정상 작동 합니다<br>
시스템 로컬이 UTF-8이 아닐 경우 한글및 특수문자 경로를 읽을 수 없습니다<br>
1. config.ini를 사용자에 맞게 수정합니다 (첫 실행시 생성됩니다)<br>
2. cmd에서 TagMetaUpdater.exe를 실행, 혹은 그냥 더블클릭해서 실행합니다<br>
<br>
<br>
<br>
<br>
<br>
<h2>config.ini 설정</h2>
selected_model = ViT<br>
사용 모델을 입력합니다, 사용 가능한 모델은 SwinV2, ConvNext, ViT 가 있습니다<br>
<br>
threshold = 0.2<br>
임계값을 설정합니다, 높을수록 정확도가 높은 태그만 예측합니다<br>
<br>
replace_tags = false<br>
기존의 태그덮어씌울지 설정합니다. false로 설정 시 기존의 태그를 유지하고 새로운 태그를 추가합니다<br>
<br>
modify_utime = false<br>
파일을 수정한 날짜를 변경합니다. false 설정시 태그를 입력해도 수정시간이 변경되지 않습니다 (앨범 날짜 관리용)<br>