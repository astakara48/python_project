### 파일 실행 순서

get_url.py -> get_content.py -> data_normalize.py -> extract_nouns.py -> get_TFIDF_top20.py -> generate_word_dict.py  
-> generator_train_model.py ->
-> generate_pikle_for_category_classification.py -> category_classification.py -> generator_test_model.py -> word_to_index_and_padding.py -> test.py -> execute.py



### File이 담당하는 part

#### 1. Data crawling

- get_url.py  ->  get_content.py 



#### 2. Data preprocessing  &  Data EDA

- data_normalize.py  ->  extract_nouns.py  -> get_TFIDF_top20.py  ->  generate_word_dict.py



#### 3. Category Classification

- generate_pikle_for_category_classification.py  ->  category_classification.py



#### 4. Training & Test Model

- getnerator_test_model.py  ->   word_to_index_and_padding.py  -> test.py  -> execute.py





### File 설명

#### 1. get_url.py

- Form Data 형식을 입력하여 빅카인즈에 입력하여 해당 형식에 맞는 기사들의 url을 가져와서 저장한다.



#### 2. get_content.py

- get_url로 만든 csv파일을 이용하여 해당 url에 접속하여 기사 내용을 긁어온다.
- 해당 url에 접속했는데 데이터가 비어있는 경우가 있어서 함수 형태로 만들고 try, exception 조건을 부여함

``` python
def insert_df(num):
    cnt_fx = num
    next_num = cnt_fx
    try:
        for i in range(cnt_fx,len(data)):
            url = df.iloc[:,-1][i]
            response = requests.get(url)
            test = response.text
            test = test.replace('false','"false"')
            dic = eval(test)
            
            # url에 접속하면 dictionary 구조
            tmp = eval(str(dic['detail']))
            data.loc[i] = [tmp['DATE'], tmp['CATEGORY_MAIN'], 
                           tmp['TMS_RAW_STREAM'], tmp['TITLE'],
                           tmp['CONTENT']]
            print(str(cnt_fx)+"번 완료")
            cnt_fx += 1
            next_num = cnt_fx+1
    except SyntaxError:
        print(next_num)
        insert_df(next_num)
```



#### 3. data_normalize

- 정규화 진행 (https://github.com/lovit/soy/blob/4c97e35cd78f2079897857c4ad4ec4a4d6a7c0f1/soy/nlp/hangle/_hangle.py)
- 조사, 부사를 제거하고 단어 통일화 작업
- 예를들어 노트북, 노트북이, 노트북을 이라는 단어들이 있다면 노트북 으로 통일화



#### 4. extract_nouns

- Okt를 이용해서 명사만 추출
- 명사 추출 결과를 열로 저장



#### 5. **generate_word_dict**

- 단어 사전 생성
- Komaran을 이용
- NA(분석 불능 범주), NR(수사), NNB(의존명사로 시작하는 단어), IC(감탄사)는 단어 사전에서 제외
- 자음 or 모음만 있는 단어 제외

``` python
    words.insert(0, ['<PAD>']) # 패딩
    words.insert(1, ['<UNK>']) # unknown 단어
    if title :
        words.insert(2, ['<S>']) # start
        words.insert(3, ['<E>']) # end
```

- 공통적으로 padding, unknown을 dict에 추가하고 타이틀에는 start, end 신호를 추가함



#### 6. generator_train_model



