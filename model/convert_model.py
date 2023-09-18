from transformers import BertModel, PreTrainedTokenizerFast, BartForConditionalGeneration
import torch
from keybert import KeyBERT
from kiwipiepy import Kiwi
from PIL import Image, ImageDraw, ImageFont
import os
from bing_image_downloader.bing_image_downloader import downloader

from bs4 import BeautifulSoup as bs
from urllib.request import urlopen, Request
from urllib.parse import quote_plus
import urllib.request
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

kiwi = Kiwi()

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
model_kobart = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')  # 글 & 이미지 중심 요약문 생성
model_kobert = BertModel.from_pretrained('skt/kobert-base-v1')  # 키워드 중심



# model_num : FE에서 받은 드롭다운 배열
def getKeyword(text, target_image_num):
  input_ids = tokenizer.encode(text)
  input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
  input_ids = torch.tensor([input_ids])


  if target_image_num == 2:  # 글 중심
    lpenalty = 2.0
    mlength = 256 #150
    minlength = 128
    nbeams = 6 #4
  else:
    lpenalty = 2.0
    mlength = 60
    minlength = 10
    nbeams = 4



  summary_text_ids = model_kobart.generate(
    input_ids=input_ids,
    bos_token_id=model_kobart.config.bos_token_id,
    eos_token_id=model_kobart.config.eos_token_id,
    length_penalty= lpenalty,
    max_length=mlength,
    min_length=minlength,
    num_beams=nbeams,
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

def searchingWords_extractor(text, k, img_type):
  nouns = noun_extractor(text)
  noun_str = ' '.join(nouns)

  kw_model = KeyBERT(model_kobert)
  keywords = kw_model.extract_keywords(noun_str, highlight=True)

  search_keywords = kw_model.extract_keywords(noun_str, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=k, diversity=0.9)
  search_keylist = list(search_keywords)
  n = len(search_keylist)

  searching_words = []  # 이미지 검색 키워드 리스트
  searching_type_words = []

  for i in range(n):
    searching_words.append(search_keylist[i][0])

  for j in searching_words:
     searching_type = j + img_type
     searching_type_words.append(searching_type)


  return searching_type_words


def final_convert(script_data, input_text, model_data, img_type):
    target_image_num = int(model_data)+1
    target_image_num_str = str(target_image_num)
    target_image = Image.open('static/background_img'+target_image_num_str+'.png')          # 배경이미지 : 클라이언트에서 넘어오는 버튼의 인덱스 값을 받아서, 배경 이미지 오픈
                                                                    
    
    fontsFolder = '/Applications/Unity/Hub/Editor/2021.3.25f1/PlaybackEngines/AndroidPlayer/SDK/platforms/android-29/data/fonts'
    selectedFont_title = ImageFont.truetype(os.path.join(fontsFolder,'NanumGothic.ttf'), 60)
    selectedFont_word = ImageFont.truetype(os.path.join(fontsFolder,'NanumGothic.ttf'), 100)
    selectedFont_content = ImageFont.truetype(os.path.join(fontsFolder,'NanumGothic.ttf'), 36)

    width, height = target_image.width, target_image.height  # (1966, 1102)
    margin = width * 0.04
    img_width, img_height = 700, 500
    title = input_text

    # 이미지 중심
    if target_image_num == 1 :
      
      #model_num = model_data                                          # model_num : FE에서 받은 드롭다운 배열
      keyword = getKeyword(script_data, target_image_num) 
      # title = getTitle
      # keyword = getKeyword(getText)
      searching_words = searchingWords_extractor(keyword, int(model_data)+2, img_type)
      print('요약문 길이: ', len(keyword))
      print('키워드: ', searching_words)

      for i in searching_words:
        query = i
        downloader.download(query, limit=1,  output_dir='./', adult_filter_off=True, force_replace=False, timeout=60)

      image_path0 = './' + searching_words[0]
      image_path1 = './' + searching_words[1]
      add_image0 = Image.open(image_path0 + '/' + os.listdir(image_path0)[0])
      add_image1 = Image.open(image_path1 + '/' + os.listdir(image_path1)[0])
      add_image0 = add_image0.resize((img_width, img_height)) 
      add_image1 = add_image1.resize((img_width, img_height)) 

      target_image.paste(im=add_image0, box=(int(margin*2.5), int(margin*3.5)))
      target_image.paste(im=add_image1, box=(int(margin*4.5+img_width), int(margin*3.5)))
      

      draw = ImageDraw.Draw(target_image)

      draw.text((margin, margin), title, fill="black", font=selectedFont_title, align='center')
      
      txt = ''
      pos = height - margin * 2.2

      while len(keyword) > 0:
          if len(keyword) > 50:
              txt = keyword[:60] + '\n'
              keyword = keyword[60:]
          else:
              txt = keyword  
              keyword = ''
          draw.text((margin, pos), txt, fill="black", font=selectedFont_content, align='left')
          
          # 텍스트를 그린 후에 높이를 업데이트
          pos += margin * 0.6

      return target_image
    


    # 글 중심
    elif target_image_num == 2 :
                     
      keyword = getKeyword(script_data, target_image_num)
      
      # 공백 제거
      sentences = keyword.strip().split('.')
      sentences = sentences[:-1]
      print(sentences)
      sentences = [i.strip() + '.' for i in sentences]
      keyword = ' '.join(sentences)
      print(keyword)
      
      searching_words = searchingWords_extractor(keyword, int(model_data),img_type)
      print('요약문 길이: ', len(keyword))
      print('키워드: ', searching_words)

      for i in searching_words:
        query = i
        downloader.download(query, limit=1,  output_dir='./', adult_filter_off=True, force_replace=False, timeout=60)

      image_path0 = './' + searching_words[0]
      add_image0 = Image.open(image_path0 + '/' + os.listdir(image_path0)[0])
      add_image0 = add_image0.resize((img_width, img_height)) 

      target_image.paste(im=add_image0, box=(int(margin), int(margin*4)))

      draw = ImageDraw.Draw(target_image)

      draw.text((margin, margin), title, fill="black", font=selectedFont_title, align='center')

      txt = ''
      pos = margin * 4

      while len(keyword) > 0:
          if len(keyword) > 10:
              txt = keyword[:35] + '\n'
              keyword = keyword[35:]
          else:
              txt = keyword  
              keyword = ''
          draw.text((margin * 2.3 + img_width, pos), txt, fill="black", font=selectedFont_content, align='center')
          
          # 텍스트를 그린 후에 높이를 업데이트
          pos += margin

      target_image.save("slide.png") # 이미지를 저장

      return target_image
      


    # 키워드 중심
    elif target_image_num == 3:
      keyword = getKeyword(script_data, target_image_num) 

      searching_words = searchingWords_extractor(keyword, int(model_data)+1, img_type)
      print('요약문 길이: ', len(keyword))
      print('키워드: ', searching_words)

      
      draw = ImageDraw.Draw(target_image)

      x_position = margin*3
      area_width = 3000 // 3


      for i, word in enumerate(searching_words):
        text_width, text_height = draw.textsize(word, font=selectedFont_word)
        print(text_width)
        x_position = (area_width // 2 - text_width/2) + i * area_width *0.9
        y_position = margin*7.5
        draw.text((x_position, y_position), word, fill="black", font=selectedFont_word)
        print("x pos: ", x_position)

      draw.text((margin, margin), title, fill="black", font=selectedFont_title, align='center')
      
      target_image.save("slide.png") # 이미지를 저장

      return target_image
       