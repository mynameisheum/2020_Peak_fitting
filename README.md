# 2020_Peak_fitting
-Deep Learning Applied to Peak Fitting of Spectroscopic Data in Frequency Domain

### XPS 배경 지식
- x선을 통해 시료에서 광전효과로 인해 나오는 전자의 운동에너지를 분석
- 이는 x축을 Binding Energy, y축을 intensity로 여러가지 peak이 섞인 그래프로 표현
- 물질 내 원소의 화학적 성질이나 전자의 상관관계를 분석
![xps ex1](https://github.com/mynameisheum/2020_Peak_fitting/blob/main/make%20Train%20data%20code/ex_picture_storage/xps%20ex.png?raw=true)

### 목표
- 기존 선행연구에서는 직접 peak를 선행적 경험에 의해 fitting하거나 area, Threshold 값등, 다양한 parameter를 필요로 했는데 neural network를 통해 단순히 데이터만으로도 peak들을 예측할수 있을까? 

### 기본 아이디어
- receptive field 역할을 갖는 Convolution neural network을 이용하면 각각의 peak을 구분할수 있지 않을까?
![xps ex2](https://github.com/mynameisheum/2020_Peak_fitting/blob/main/make%20Train%20data%20code/ex_picture_storage/xps%20ex2.png?raw=true)

### 느낀점
-데이터를 밀어넣는 방법엔 한계가 있다.

### 문제점
-

