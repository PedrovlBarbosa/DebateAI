import os
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch
import json
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import random
import string
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips
import gradio as gr
from os.path import exists

# Api keys
def setEnvVariables(openAi, huggingFace):
  os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingFace
  os.environ['OPENAI_API_KEY'] = openAi

#Escreva o primeiro personagem em charA e o segundo em charB
def setCharacters(fChar, sChar):
  charA = fChar.title()
  charB = sChar.title()

#   listA = list(charA)
#   listB = list(charB)

#   for i in range(len(listA)):
#     if i == 0:
#       if ord(listA[0]) > 90:
#         listA[0] = chr(ord(listA[0]) - 32)
#     if listA[i] == " " and i + 1 < len(listA):
#       if ord(listA[i+1]) > 90:
#         listA[i+1] = chr(ord(listA[i+1]) - 32)

#   for i in range(len(listB)):
#     if i == 0:
#       if ord(listB[0]) > 90:
#         listB[0] = chr(ord(listB[0]) - 32)
#     if listB[i] == " " and i + 1 < len(listB):
#       if ord(listB[i+1]) > 90:
#         listB[i+1] = chr(ord(listB[i+1]) - 32)

#   charA = ''.join(listA)
#   charB = ''.join(listB)
  return charA, charB

#choosing processing device
def defineProcessingDevice():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  return device

# Geração de imagens (PESADO, se não rodar use a opção de baixo)
# load both base & refiner
def highQualityRendering(device, charA, charB):
  if device == "cuda":
    base = DiffusionPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
  else:
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, variant="fp16", use_safetensors=True
    )
  base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
  base.to(device)

  if device == "cuda":
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
  else:
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float32,
        use_safetensors=True,
        variant="fp16",
    )
  refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
  refiner.to(device)

  # Define how many steps and what % of steps to be run on each experts (80/20) here
  n_steps = 40
  high_noise_frac = 0.8


  def generateCharA(char):
      prompt = char+" closeup, turning to the right, in a black and white background, highly-detailed"
      # run both experts
      image = base(
          prompt=prompt,
          num_inference_steps=n_steps,
          denoising_end=high_noise_frac,
          output_type="latent",
      ).images
      image = refiner(
          prompt=prompt,
          num_inference_steps=n_steps,
          denoising_start=high_noise_frac,
          image=image,
      ).images[0]
      image.save("charA.png")
      return image

  def generateCharB(char):
      prompt = char+" closeup, turning to the left, in a black and white background, highly-detailed"
      # run both experts
      image = base(
          prompt=prompt,
          num_inference_steps=n_steps,
          denoising_end=high_noise_frac,
          output_type="latent",
      ).images
      image = refiner(
          prompt=prompt,
          num_inference_steps=n_steps,
          denoising_start=high_noise_frac,
          image=image,
      ).images[0]
      image.save("charB.png")
      return image

  generateCharA(charA)
  generateCharB(charB)

#Geração de imagens (LEVE)
def lowQualityRendering(device, charA, charB):
  model_id = "runwayml/stable-diffusion-v1-5"
  if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
  else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
  pipe = pipe.to(device)

  def generateCharA(char):
    prompt = char+" closeup, turning to the right, black and white background, highly-detailed"
    image = pipe(prompt).images[0]
    image.save("charA.png")

  def generateCharB(char):
    prompt = char+" closeup, turning to the left, black and white background, highly-detailed"
    image = pipe(prompt).images[0]
    image.save("charB.png")

  generateCharA(charA)
  generateCharB(charB)

#Gerando a descrição textual dos áudios
def setupDebate(charA, charB):
  def generateDebate(characterA, characterB):
    print(characterA+" "+characterB+"!!!!")
    template = """
    ChatGPT, generate a textual arguing between {characterA} and {characterB}, dividing it in seven small frases to each one and formatting it
    in a json structure where {characterA} and {characterB} are both keys that holds their respective frases.
    After that, answer to me if {characterA} and {characterB} are male or female formatting your answer in a dictionary
    json structure, where {characterA} and {characterB} are both keys holding the answers, if male valuate to 0 an if famale valuate to 1.
    """
    prompt = PromptTemplate(template=template, input_variables=["characterA","characterB"])
    llm = OpenAI(temperature=0.9)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    test = {'characterA':characterA , 'characterB':characterB}
    Story = llm_chain.run(test)
    return Story
  answer = generateDebate(charA, charB)

  answerToJson = '{ "argue": '+answer+'}'

  for i in range(len(answerToJson)):
    if answerToJson[i] == '}':
      part1 = answerToJson[0: i+1]
      part2 = ', "gender": '+answerToJson[i+1: len(answerToJson)]
      answerToJson = part1+part2
      break

  print(answerToJson)
  json_answer = json.loads(answerToJson)

  frasesA = json_answer["argue"][charA]
  frasesB = json_answer["argue"][charB]

  aGender = json_answer["gender"][charA]
  bGender = json_answer["gender"][charB]
  return aGender, bGender, frasesA, frasesB

#generating audio
def generateAudios(device, aGender, bGender, frasesA, frasesB):
  processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
  model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
  vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
  embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

  speakers = {
      'awb': 0,     # Scottish male
      'bdl': 1138,  # US male
      'clb': 2271,  # US female
      'jmk': 3403,  # Canadian male
      'ksp': 4535,  # Indian male
      'rms': 5667,  # US male
      'slt': 6799   # US female
  }

  def save_text_to_speech(text, speaker, fileName):
    # preprocess text
    inputs = processor(text=text, return_tensors="pt").to(device)
    speaker_embeddings = torch.tensor(embeddings_dataset[speaker]["xvector"]).unsqueeze(0).to(device)
    # generate speech with the models
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    output_filename = f"{fileName}.mp3"
    # save the generated speech to a file with 16KHz sampling rate
    sf.write(output_filename, speech.cpu().numpy(), samplerate=16000)
    # return the filename for reference
    return output_filename

  if aGender == 0 and bGender == 0:
    for i in range(len(frasesA)):
      save_text_to_speech(frasesA[i], speakers["rms"], "a"+str(i))
    for i in range(len(frasesB)):
      save_text_to_speech(frasesB[i], speakers["jmk"], "b"+str(i))
  elif aGender == 1 and bGender == 1:
    for i in range(len(frasesA)):
      save_text_to_speech(frasesA[i], speakers["slt"], "a"+str(i))
    for i in range(len(frasesB)):
      save_text_to_speech(frasesB[i], speakers["clb"], "b"+str(i))
  else:
    if aGender == 0:
      for i in range(len(frasesA)):
        save_text_to_speech(frasesA[i], speakers["rms"], "a"+str(i))
      for i in range(len(frasesB)):
        save_text_to_speech(frasesB[i], speakers["slt"], "b"+str(i))
    else:
      for i in range(len(frasesA)):
        save_text_to_speech(frasesA[i], speakers["slt"], "a"+str(i))
      for i in range(len(frasesB)):
        save_text_to_speech(frasesB[i], speakers["rms"], "b"+str(i))
  audioA = ["app\a0.mp3", "app\a1.mp3", "app\a2.mp3", "app\a3.mp3", "app\a4.mp3", "app\a5.mp3", "app\a6.mp3"]
  audioB = ["app\b0.mp3", "app\b1.mp3", "app\b2.mp3", "app\b3.mp3", "app\b4.mp3", "app\b5.mp3", "app\b6.mp3"]
  return audioA, audioB

videosA = []
videosB = []

def buildVideo(charA, charB, temp, openAi_key, hugginFace_key):
  setEnvVariables(openAi_key, hugginFace_key)
  fChar, sChar = setCharacters(charA, charB)
  pDevice = defineProcessingDevice()
  aGender, bGender, frasesA, frasesB = setupDebate(fChar, sChar)
  audioA, audioB = generateAudios(pDevice, aGender, bGender, frasesA, frasesB)
  #highQualityRendering(pDevice, fChar, sChar)  # Escolha qual renderização
  lowQualityRendering(pDevice, fChar, sChar)    # você prefere

  for i in range(0, len(audioA)):
      audio = AudioFileClip(audioA[i])
      vidImgA = ImageClip("app\charA.png").set_duration(audio.duration)
      vidImgA = vidImgA.set_audio(audio)
      videosA.append(vidImgA)

  for i in range(0, len(audioB)):
      audio = AudioFileClip(audioB[i])
      vidImgB = ImageClip("app\charB.png").set_duration(audio.duration)
      vidImgB = vidImgB.set_audio(audio)
      videosB.append(vidImgB)

  all_vid = []

  for i in range(0, len(videosA)):
      all_vid.append(videosA[i])
      all_vid.append(videosB[i])

  full_vid = concatenate_videoclips(all_vid)
  full_vid = full_vid.speedx(factor=1)
  full_vid.write_videofile("app\output.mp4", fps=24)
  return full_vid

def video_generator(Personagem_A, Personagem_B, Temperatura, openAI_key, hugginFace_key):
    buildVideo(Personagem_A, Personagem_B, Temperatura, openAI_key, hugginFace_key)
    file_exists = exists("app\output.mp4")
    while not file_exists:
      file_exists = exists("app\output.mp4")
    mp4 = os.path.join(os.path.abspath(''), "app\output.mp4")  # Video
    return mp4

demo = gr.Interface(
    video_generator,
    inputs=["text", "text", gr.Slider(0, 100), "text", "text"],
    outputs=gr.Video(),
    cache_examples=True
    )
demo.launch(debug=True)