TagMetaUpdater
=
<br>
<img src="img\example.PNG"><br>
이미지의 태그를 인식해서 XMPmeta데이터를 수정하는 프로그램 입니다<br>
시스템 로컬을 UTF-8로 변경해야 정상 작동 합니다<br>
시스템 로컬이 UTF-8이 아닐 경우 한글및 특수문자 경로를 읽을 수 없습니다<br>
<br>
첫 실행시 https://huggingface.co/SmilingWolf 에서 모델을 받아옵니다<br>
모델은 캐시에 저장되기 때문에 webui에 있는 wd14 tagger를 사용해봤다면 이 과정은 생략될거임<br>
<br>
<br>
이미지를 인식하는 코드와 모델은 https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags 에서 가져왔으며<br>
그외의 코드는 chatGPT가 작성하였습니다<br>
<br>
<br>
<br>
<h1>사용방법</h1>
1. config.ini를 사용자에 맞게 수정합니다
2. cmd에서 TagMetaUpdater.exe를 실행, 혹은 그냥 더블클릭해서 실행합니다