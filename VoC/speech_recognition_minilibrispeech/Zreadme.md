# 模型从数据到搭建到训练全过程说明

## 🚩 step1:模型数据预处理说明：
> 数据是模型的粮食，是生命，其重要性不言而喻。
> 现实中我拿到的数据是凌乱的，都需要我们进行处理，把他整理为我们任务需要的格式，
> 不任务格式不同，需要根据任务情况和用到的深度学习框架来确定具体任务的数据格式
- ‍🚴‍<font style="color:red">这个一部分是根据使用的框架和任务自行编码处理实现的。</font>
  - 语音识别+SpeechBrain数据格式：
  ```
  {
  "1867-154075-0032": {
    "wav": "{data_root}/LibriSpeech/train-clean-5/1867/154075/1867-154075-0032.flac",
    "length": 16.09,
    "words": "AND HE BRUSHED A HAND ACROSS HIS FOREHEAD AND WAS INSTANTLY HIMSELF CALM AND COOL VERY WELL THEN IT SEEMS I'VE MADE AN ASS OF MYSELF BUT I'LL TRY TO MAKE UP FOR IT NOW WHAT ABOUT CAROLINE"
  },
  "1867-154075-0001": {
    "wav": "{data_root}/LibriSpeech/train-clean-5/1867/154075/1867-154075-0001.flac",
    "length": 14.9,
    "words": "THAT DROPPED HIM INTO THE COAL BIN DID HE GET COAL DUST ON HIS SHOES RIGHT AND HE DIDN'T HAVE SENSE ENOUGH TO WIPE IT OFF AN AMATEUR A RANK AMATEUR I TOLD YOU SAID THE MAN OF THE SNEER WITH SATISFACTION"
  },
  "1867-154075-0028": {
    "wav": "{data_root}/LibriSpeech/train-clean-5/1867/154075/1867-154075-0028.flac",
    "length": 16.41,
    "words": "MY NAME IS JOHN MARK I'M DOONE SOME CALL ME RONICKY DOONE I'M GLAD TO KNOW YOU RONICKY DOONE I IMAGINE THAT NAME FITS YOU NOW TELL ME THE STORY OF WHY YOU CAME TO THIS HOUSE OF COURSE IT WASN'T TO SEE A GIRL"
  },
  }
  ```
## 🚩 step2: 训练词token
> 一个语音识别系统能识别的语音内容是有他所包含的词库所决定的，所以在构建语音识别系统之前需要构建一个该系统的词库，把可以识别的词都包含进来，词库的内容丰富程度
> 决定了语音识别系统能识别的内容。
- 🚴 ‍<font style="color:red">这一部分是根据训练集的词汇来构建我们的词库的，主要分为如下三种方式:</font>
  - 使用单个字母(char)，单个拼音（这种方式词库小，但是输入内容多，输出的也多，不方便合理）
  - 使用words(unigram)，单个词（这种方式减少了输入输出，但是不利于组合的词语（比如：机器学习）的归纳识别，对一些少见词的学习效果也会差）
  - 使用分词（SentencePiece，wordpiece）技术，把词库的内容按不同的粒度分为词，词组等内容
      - 基于规则的分词： spaCy    
        - grammar rules, spaces, and punctuation
        
      - BPE:把单词都拆分成字母，每次都找出相邻单元出现次数最多的进行拼接
           
           ```
          原始词表 {'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3, 'l o w </w>': 5}
          出现最频繁的序列 ('s', 't') 9
          合并最频繁的序列后的词表 {'n e w e st </w>': 6, 'l o w e r </w>': 2, 'w i d e st </w>': 3, 'l o w </w>': 5}
          出现最频繁的序列 ('e', 'st') 9
          合并最频繁的序列后的词表 {'l o w e r </w>': 2, 'l o w </w>': 5, 'w i d est </w>': 3, 'n e w est </w>': 6}
          出现最频繁的序列 ('est', '</w>') 9
          合并最频繁的序列后的词表 {'w i d est</w>': 3, 'l o w e r </w>': 2, 'n e w est</w>': 6, 'l o w </w>': 5}
          出现最频繁的序列 ('l', 'o') 7
          合并最频繁的序列后的词表 {'w i d est</w>': 3, 'lo w e r </w>': 2, 'n e w est</w>': 6, 'lo w </w>': 5}
          出现最频繁的序列 ('lo', 'w') 7
          合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'n e w est</w>': 6, 'low </w>': 5}
          出现最频繁的序列 ('n', 'e') 6
          合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'ne w est</w>': 6, 'low </w>': 5}
          出现最频繁的序列 ('w', 'est</w>') 6
          合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'ne west</w>': 6, 'low </w>': 5}
          出现最频繁的序列 ('ne', 'west</w>') 6
          合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'newest</w>': 6, 'low </w>': 5}
          出现最频繁的序列 ('low', '</w>') 5
          合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'newest</w>': 6, 'low</w>': 5}
          出现最频繁的序列 ('i', 'd') 3
          合并最频繁的序列后的词表 {'w id est</w>': 3, 'newest</w>': 6, 'low</w>': 5, 'low e r </w>': 2}
          ```
        
      - wordpiece：
        - 准备足够大的训练语料
        - 确定期望的subword词表大小
        - 将单词拆分成字符序列
        - 基于第3步数据训练语言模型
        - 从所有可能的subword单元中选择加入语言模型后能最大程度地增加训练数据概率的单元作为新的单元  
        - 重复第5步直到达到第2步设定的subword词表大小或概率增量低于某一阈值
        
      - ULM(Unigram Language Model)
          -  准备足够大的训练语料
          -  确定期望的subword词表大小
          -  给定词序列优化下一个词出现的概率
          -  计算每个subword的损失
          -  基于损失对subword排序并保留前X%。为了避免OOV，建议保留字符级的单元
          -  重复第3至第5步直到达到第2步设定的subword词表大小或第5步的结果不再变化
  
  > 这三种方式一般选用三种方式来构建词库能获得更好的效果
  - [分词代码加技术讲解教程](https://colab.research.google.com/drive/12yE3myHSH-eUxzNM0-FLtEOhzdQoLYWe?usp=sharing#scrollTo=0v2mq9wwfBeV)
  - 这里使用 [sentencepiece](https://github.com/google/sentencepiece) 工具来做，可用的方法为ULM,BPE，实现的分词后的效果如下:
    ```
    <unk>	0
    ▁THE	-3.24574
    S	-3.36942
    ED	-3.84667
    ▁	-3.9181
    E	-3.92003
    ▁AND	-3.92311
    ▁A	-3.97345
    ▁TO	-4.00457
    ▁OF	-4.0811
    T	-4.10382
    D	-4.2464
    ING	-4.35973
    ▁IN	-4.40413
    ▁HE	-4.52177
    Y	-4.62782
    ▁I	-4.64771
    ▁WAS	-4.70565
    M	-4.75599
    N	-4.815
    ▁THAT	-4.83189
    ```


## 🚩 step3:训练语言模型：
> 语言模型在语音识别模型中的作用是辅助判断语音识别的文字结果，可以实现一些纠错等功能，进一步提高语音识别的准确率
- 使用大规模语料库训练的模型直接作为语言模型
- 使用要训练的数据训练一个语言模型，一般效果不好
- 在已有的大规模预料库的模型基础上，使用我们的数据进行微调获得一个好滴效果