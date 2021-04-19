# human-removal


## Processing
mask-rcnn을 이용해서 동영상에서 사람들을 모두 검출합니다.

0.5초 단위로 프레임을 분할해서 사람들을 모두 검출합니다.

프레임들을 ORB feature matching을 통하여 투명 공간에 다음 프레임의 이미지를 끼워 넣습니다.

끼워진 영역에 남아있는 투명 공간은 edge-connect를 활용하여 마저 Inpainting 합니다.

## Input

![1](https://user-images.githubusercontent.com/17982163/115174783-d9be4a80-a104-11eb-8f7c-ba0daf1986be.gif)

## Output

![results1](https://user-images.githubusercontent.com/17982163/115174470-4127ca80-a104-11eb-8e19-ec5a19545360.png)

##Reference 

https://github.com/knazeri/edge-connect
