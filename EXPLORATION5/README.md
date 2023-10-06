# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 박태하
- 리뷰어 : 김지원


# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
    - **인물이 있는 다양한 구도의 사진들과 고양이 사진들로 하여금 segmentation 시도를 하였다**
    - **또한 백그라운드를 합성하는 사진까지 잘 만들었다**
     ```python3
    def combine_images(person_img_path, background_img_path):
    # 사람 이미지에서 배경 제거
    img_orig = cv2.imread(person_img_path)
    
    model_dir = os.getenv('HOME')+'/aiffel/human_segmentation/models'
    model_file = os.path.join(model_dir, 'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')
    
    model = semantic_segmentation()
    model.load_pascalvoc_model(model_file)
    _, output = model.segmentAsPascalvoc(person_img_path)
    
    seg_color = (128, 128, 192)
    seg_map = np.all(output == seg_color, axis=-1)
    
    img_mask = seg_map.astype(np.uint8) * 255
    img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    img_person_only = cv2.bitwise_and(img_orig, img_mask_color)

    # 새로운 배경 이미지와 합치기
    img_background = cv2.imread(background_img_path)
    img_background = cv2.resize(img_background, (img_orig.shape[1], img_orig.shape[0]))  # 사람 이미지와 같은 크기로 리사이즈
    img_background_masked = cv2.bitwise_and(img_background, cv2.bitwise_not(img_mask_color))
    combined_img = cv2.add(img_person_only, img_background_masked)

    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    plt.show()
    ```
    - **본 function은 배경 크로마키 코드이다. 함수화와 함께 잘 짰다 (근거)**
    
- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    - **함수화 파트 이전에 코멘트가 많이 있었다**
    - **함수화 떄는 코멘트가 없거나 부족하였다**
    ```python3
    # True과 False인 값을 각각 255과 0으로 바꿔줍니다
    img_mask = seg_map.astype(np.uint8) * 255
    # 255와 0을 적당한 색상으로 바꿔봅니다
    color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)
    # 원본 이미지와 마스트를 적당히 합쳐봅니다
    # 0.6과 0.4는 두 이미지를 섞는 비율입니다.
    img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.4, 0.0)
    ```   
    
    ```python3
    def cat_segmentation2(img_path):
    img_orig = cv2.imread(img_path)
    {...}
    model = semantic_segmentation()
    model.load_pascalvoc_model(model_file)
    segvalues, output = model.segmentAsPascalvoc(img_path)
    ```
        
- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    - **문제점의 이해도가 높았다. 다양한 구도 혹은 사진 다양성을 고려하여 시도를 많이 하였다.
    - **Comment 중 하나를 발췌해왔다.**
    ```
    사람이 너무 작아서 사람은 분류는 되지만 같이 blur처리가 되어 버린다.
    ```
        
- [ ]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    - **네, 회고 잘 작성되었습니다.**
    ```
    평소에 사진에 관심이 많아 과제 자체를 하는 것이 재밌었다. 세그멘테이션에 대해 개념적인 부분만 알고 어떻게 활용하는지는 몰랐는데 이번 학습을 통해 확실하게 알 수 있었다. 블러 처리가 생각보다 잘 안되고 잘 되더라고 테두리 부분이 잘 처리가 안 되었던 것 같다. 사진 속 대상의 크기, 입고있는 옷들에 영향을 크게 받는 것 같다.
    ```
    - **또한 rooms for improvment 도 잘 나타나있었다**
    ```
    - 3D Camera 활용하기
    스테레오 비전: 두 개의 카메라를 사용하여 물체의 깊이를 측정한다. 두 카메라 간의 거리 차이를 이용하여 3D 정보를 추출한다.
    ToF방식: 레이저를 사용하여 물체까지의 거리를 측정한다. 이 방식은 빠르고 정확한 깊이 정보를 제공한다.
    DeepLab 모델의 Semantic Segmentation 결과와 3D 카메라의 깊이 정보를 결합하여 보다 정확한 세그멘테이션을 구현할 수 있다. 깊이 정보를 활용하면 물체와 배경의 경계를 더 명확하게 구분할 수 있다.
    ```
        
- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    - **함수로 잘 짰습니다. 하지만 가독성이 더 좋았음 합니다.**
    - **함수에 여러가지 기능이 있는것 같았습니다.**

# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```