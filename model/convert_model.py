from transformers import BertModel, PreTrainedTokenizerFast, BartForConditionalGeneration
import torch
from keybert import KeyBERT
from kiwipiepy import Kiwi

from PIL import Image, ImageDraw, ImageFont
import os
import shutil
from bing_image_downloader.bing_image_downloader import downloader

from bs4 import BeautifulSoup as bs
from urllib.request import urlopen, Request
from urllib.parse import quote_plus
import urllib.request
import matplotlib.pyplot as plt

kiwi = Kiwi()

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
model_kobart = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')  # 이미지 중심
model_kobert = BertModel.from_pretrained('skt/kobert-base-v1')  # 키워드 중심

# 모델선택용 배열 생성 [0:글, 1:이미지, 2:키워드]
model_image = model_kobart
model_text = model_kobert
model_keyword = True   # 키워드용
modelSET = [model_text, model_image, model_keyword]  

# model_num : FE에서 받은 드롭다운 배열
def getKeyword(text):
  input_ids = tokenizer.encode(text)
  input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
  input_ids = torch.tensor([input_ids])

  summary_text_ids = model_kobart.generate(
    input_ids=input_ids,
    bos_token_id=model_kobart.config.bos_token_id,
    eos_token_id=model_kobart.config.eos_token_id,
    length_penalty=2.0,
    max_length=60,
    min_length=10,
    num_beams=4,
  )

  keyword = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)

  return keyword

def noun_extractor(text):
  results = []
  result = kiwi.analyze(text)
  for token, pos, _, _ in result[0][0]:
      if len(token) != 1 and pos.startswith('N') or pos.startswith('SL'):
          results.append(token)
          
  return results

def searchingWords_extractor(text):
  nouns = noun_extractor(text)
  noun_str = ' '.join(nouns)

  kw_model = KeyBERT(model_kobert)
  keywords = kw_model.extract_keywords(noun_str, highlight=True)

  search_keywords = kw_model.extract_keywords(noun_str, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=2, diversity=0.9)
  search_keylist = list(search_keywords)
  n = len(search_keylist)

  searching_words = []  # 이미지 검색 키워드 리스트

  for i in range(n):
    searching_words.append(search_keylist[i][0])

  return searching_words


def final_convert(script_data, input_text, model_data):
    target_image_num = int(model_data)+1
    target_image_num_str = str(target_image_num)
    target_image = Image.open('static/background_img'+target_image_num_str+'.png')          # 배경이미지 : 클라이언트에서 넘어오는 버튼의 인덱스 값을 받아서, 배경 이미지 오픈
                                                                     # 배경이미지 및  위치 배정 함수를 새로 만들어야 함.
    
    print("이미지 숫자 디버깅중: "+target_image_num_str)
    # 모델에 따라 다른 배치 구성하기
    # if model_data == 0 :
    #   width, height = target_image.width, target_image.height  # (1966, 1102)
    #   margin = width * 0.04
       

    width, height = target_image.width, target_image.height  # (1966, 1102)
    margin = width * 0.04
    img_width, img_height = 700, 500

    # getTitle: 받은 목차, getText: 받은 대본

    title = input_text
    #model_num = model_data                                          # model_num : FE에서 받은 드롭다운 배열
    keyword = getKeyword(script_data) 
    # title = getTitle
    # keyword = getKeyword(getText)
    searching_words = searchingWords_extractor(keyword)
    print('요약문 길이: ', len(keyword))
    print('키워드: ', searching_words)

    # 크롤링
    for i in searching_words:
        query = i
        downloader.download(query, limit=1,  output_dir='./', adult_filter_off=True, force_replace=False, timeout=60)


    add_image0 = Image.open('./' + searching_words[0] + '/image_1.jpg')
    add_image1 = Image.open('./' + searching_words[1] + '/image_1.jpg')
    add_image0 = add_image0.resize((img_width, img_height)) 
    add_image1 = add_image1.resize((img_width, img_height)) 

    target_image.paste(im=add_image0, box=(int(margin*2.5), int(margin*3)))
    target_image.paste(im=add_image1, box=(int(margin*4.5+img_width), int(margin*3)))

    fontsFolder = '/Applications/Unity/Hub/Editor/2021.3.25f1/PlaybackEngines/AndroidPlayer/SDK/platforms/android-29/data/fonts'
    selectedFont_title = ImageFont.truetype(os.path.join(fontsFolder,'NanumGothic.ttf'), 60)
    selectedFont_content = ImageFont.truetype(os.path.join(fontsFolder,'NanumGothic.ttf'), 36)


    draw = ImageDraw.Draw(target_image)

    draw.text((margin, margin), title, fill="black", font=selectedFont_title, align='center')
    if len(keyword) > 55:
        txt = keyword[:60] + '\n' + keyword[60:]
        pos = height - margin * 2.6
    else:
        pos = height - margin * 2.2
    draw.text((margin, pos), txt, fill="black", font=selectedFont_content, align='left')
    # target_image.save("slide.png") # 이미지를 저장
    

    return target_image
