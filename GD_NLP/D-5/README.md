# AIFFEL_Quest
The process of archiving quests performed during the iPel process

# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 임정훈
- 리뷰어 : 김서연


# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    ```python
    def preprocess_sentence(sentence):
        
        sentence = sentence.lower().strip()
    
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = re.sub(r"[^a-zA-Z?.!가-힣ㄱ-ㅎㅏ-ㅣ]+", " ", sentence)
    
        sentence = sentence.strip()
        
        return sentence
    ```
    - 텍스트 전처리 함수를 잘 작성했다.
    - 번역기 모델이 잘 작동하지 않은 것으로 보인다(ㅠㅠ).
    
- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    ```python3
    def generate_masks(inp, tar):
        # 인코더 패딩 마스크
        enc_padding_mask = create_padding_mask(inp)
    
        # 디코더의 두 번째 어텐션 모듈을 위한 패딩 마스크
        dec_padding_mask = create_padding_mask(inp)
    
        # 디코더의 첫 번째 어텐션 모듈을 위한 룩-어헤드 마스크
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
        return enc_padding_mask, combined_mask, dec_padding_mask
    
    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # 추가 차원을 넣어서 shape을 맞추기 위해 expand_dims를 사용
        return seq[:, tf.newaxis, tf.newaxis, :]
    
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        # (seq_len, seq_len) 크기의 상삼각행렬 생성
        return mask
    ```
    - 위 코드와 같이 주석이 잘 작성된 부분이 있으나 일부 주석이 없는 코드도 있다. 특히 실제 학습이 진행되는 코드에 주석이 추가되면 더 좋을 것 같다.
               
- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 학습을 진행하는 단계에서 발생한 에러에 대한 분석이 필요해보인다.
        
- [ ]  **4. 회고를 잘 작성했나요?**
      <img width="891" alt="스크린샷 2023-11-17 오후 5 47 17" src="https://github.com/ScientistLim/AIFFEL_Quest/assets/112914475/2bf7ba84-346f-4b0e-8d5d-d317c87a7ad1">
     - 회고가 잘 작성되었다.
     - 어떤 에러가 왜 발생했는지 적어보면 좋을 것 같다.
        
- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - 코드가 간결하게 작성되었다.


# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
