# 차량지능기초 전반기 프로젝트
- 학번: 20171718
- 이름: 한정택

## 개요
2021학년도 소프트웨어학부 차량지능기초 수업의 전반기 프로젝트입니다.\
Mediapipe 와 scikit-learn 프레임워크를 활용하여
운전자가 주행보조기능에 의존하지 않는지 확인해주는 기능을 구현하고자합니다.

## Environment
- Interpreter: Python 3.9

## Dependency
- **mediapipe**: Holistic solution을 활용하여 운전자의 Face와 Pose Feature 를 추출할 떄 활용합니다.
- **cv2**: 인지 결과, 커스텀 모델을 활용한 추론결과를 영상에 합성하는 등의 처리를 위해 사용합니다.
- **pandas**: dataset이 담긴 CSV 파일을 컨트롤하기위해 활용합니다.
- **scikit-learn**: 각종 Regression 모델을 활용하여 간단한 분류기를 만들기 위해 활용합니다.
- **numpy**: mediapipe 에서 인식한 데이터의 구조화에 사용됩니다.

## File
아래와 같은 순서로 파일을 실행하여 결과에 도달할 수 있습니다.

1. **[InitializeCSV.py](https://github.com/smart8612/CarInteligence-Mediapipe/blob/master/InitializeCSV.py )**\
   사용자의 Pose와 Face Detection dataset을 저장할 CSV 파일을 생성합니다.\
   사용자가 인식되는 상태에서 'q' 버튼을 클릭하면 프로젝트 디렉토리에 'coords.csv' 파일이 생성됩니다. \
   
   
2. **[AddClassToCSV.py](https://github.com/smart8612/CarInteligence-Mediapipe/blob/master/AddClassToCSV.py )**\
   처음에 실행하면 class 이름을 지정하라는 프롬프트 메시지가 뜹니다.\
   이때 앞으로 취할 Pose 와 Face 형태에 대하여 어떤 의미를 갖고 있는지 적어주면 됩니다.\
   이후 카메라를 보면서 관련 행동을 취하면 CSV 파일에 feature 데이터가 추가됩니다. \
   다른 행동을 등록하려면 이 파일을 재실행하여 같은방식으로 등록하면 됩니다.\
   마찬가지로 'q' 버튼을 클릭하면 종료됩니다.
   
   
3. **[TrainCustomModel.py](https://github.com/smart8612/CarInteligence-Mediapipe/blob/master/TrainCustomModel.py )**\
   'coords.csv' 파일을 불러와서 LogisticRegression 모델을 학습시키고\
   결과 분류기를 파이썬의 pickle 을 활용하여 binary 파일 형태로 export 합니다.\
   
   
4. **[DetectWithCustomModel.py](https://github.com/smart8612/CarInteligence-Mediapipe/blob/master/DetectWithCustomModel.py )**\
   생성된 "DriverRecognition.pkl" 분류기를 활용하는 단계입니다.\
   Mediapipe 를 통해 실시간 인식되는 운전자의 자세를 분류기에게 전달하여 \
   어떤 class 에 매칭되는지 추론하게됩니다. \
   결과는 opencv를 활용하여 사용자의 좌측어깨와 우측어깨에 매칭된 class 와 확률을 출력시켜줍니다.
