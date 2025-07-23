from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Optional, List

class _Quantidade(BaseModel):
    valor: Optional[float] = None
    unidade: Optional[str] = None

class _Dimensoes(BaseModel):
    valor: Optional[str] = None
    unidade: Optional[str] = None

    @model_validator(mode='before')
    def input_normalize(cls, data):
        if isinstance(data, dict):
            value = data.get("valor")
            if value is not None:
                data['valor'] = str(value)
        return data

class _Peso(BaseModel):
    valor: Optional[float] = None
    unidade: Optional[str] = None

class _Caracteristicas(BaseModel):
    dimensoes: Optional[_Dimensoes] = Field(default_factory=_Dimensoes)
    peso: Optional[_Peso] = Field(default_factory=_Peso)
    volume_por_unidade: Optional[_Quantidade] = Field(default_factory=_Quantidade)

    @field_validator('dimensoes', mode='before', check_fields=False)
    def ensure_dimensoes(cls, v):
        if v is None:
            return _Dimensoes()
        return v

    @field_validator('peso', mode='before', check_fields=False)
    def ensure_peso(cls, v):
        if v is None:
            return _Peso()
        return v

    @field_validator('volume_por_unidade', mode='before', check_fields=False)
    def ensure_volume_por_unidade(cls, v):
        if v is None:
            return _Quantidade()
        return v

class _Embalagem(BaseModel):
    tipo: Optional[str] = None
    quantidade: Optional[_Quantidade] = Field(default_factory=_Quantidade)

    @field_validator('quantidade', mode='before', check_fields=False)
    def ensure_quantidade(cls, v):
        if v is None:
            return _Quantidade()
        return v

class _DetalhesExtraidos(BaseModel):
    embalagem: Optional[_Embalagem] = Field(default_factory=_Embalagem)
    caracteristicas: Optional[_Caracteristicas] = Field(default_factory=_Caracteristicas)

    @model_validator(mode='before')
    def ensure_structure(cls, data):
        if data is None:
            return {}
        if data.get("embalagem") is None:
            data["embalagem"] = {}
        if data.get("caracteristicas") is None:
            data["caracteristicas"] = {}
        return data

class _InformacoesAdicionais(BaseModel):
    marca: Optional[str] = None
    origem: Optional[str] = None
    categoria: Optional[List[str]] = Field(default_factory=list)

class _DescricaoEnriquecida(BaseModel):
    produto_base: Optional[str] = None
    produto_detalhado: Optional[str] = None
    detalhes_extraidos: Optional[_DetalhesExtraidos] = Field(default_factory=_DetalhesExtraidos)
    informacoes_adicionais: Optional[_InformacoesAdicionais] = Field(default_factory=_InformacoesAdicionais)

class _Descricao(BaseModel):
    original: str
    enriquecida: Optional[_DescricaoEnriquecida] = Field(default_factory=_DescricaoEnriquecida)
    tags: Optional[List[str]] = Field(default_factory=list)

class _ValorUnitario(BaseModel):
    valor: Optional[float] = None
    moeda: Optional[str] = "BRL"

class _Produto(BaseModel):
    id_item: int
    id_nfe: Optional[str] = None
    ncm: Optional[str] = None
    gtin: Optional[str] = None
    descricao: _Descricao
    unidade_comercializacao: Optional[str] = None
    quantidade_comercializada: _Quantidade = Field(default_factory=_Quantidade)
    valor_unitario: _ValorUnitario

class NFeMinerJSONValidator(BaseModel):
    produto: _Produto

class NFeMinerBaseGenerateModel(ABC):

	instruction = '''Você é um modelo de linguagem treinado para processar dados de notas fiscais eletrônicas (NFEs) em português. Sua tarefa é analisar a descrição do item e outros campos fornecidos, extrair informações estruturadas e retornar um JSON formatado conforme uma estrutura predefinida.

**Regras importantes:**
1. As notas fiscais estão em português.
2. O JSON de saída deve ser formatado conforme a estrutura detalhada abaixo.
3. Certifique-se de separar números de suas unidades para campos como quantidade, peso, volume, etc.
4. Utilize informações presentes na descrição para preencher campos específicos no JSON, enriquecendo os dados sempre que possível.
5. Se um campo não puder ser inferido, deixe-o vazio ou use `null`.

---

**Estrutura esperada do JSON de saída:**

```json
{
  "produto": {
    "id_item": "<ID único do item>",
    "id_nfe": "<ID único da nota fiscal>",
    "ncm": "<Código NCM>",
    "gtin": "<Código GTIN>",
    "descricao": {
      "original": "<Descrição original extraída da nota fiscal>",
      "enriquecida": {
        "produto_base": "<Nome base do produto>",
        "produto_detalhado": "<Descrição detalhada inferida do produto>",
        "detalhes_extraidos": {
          "embalagem": {
            "tipo": "<Tipo de embalagem (ex.: caixa, saco, garrafa)>",
            "quantidade": {
              "valor": "<Número>",
              "unidade": "<Unidade (ex.: unidades, litros, kg)>"
            }
          },
          "caracteristicas": {
            "dimensoes": {
              "valor": "<Número>",
              "unidade": "<Unidade (ex.: cm, m, litros)>"
            },
            "peso": {
              "valor": "<Número>",
              "unidade": "<Unidade (ex.: gramas, kg)>"
            },
            "volume_por_unidade": {
              "valor": "<Número>",
              "unidade": "<Unidade (ex.: mL, litros)>"
            }
          }
        },
        "informacoes_adicionais": {
          "marca": "<Marca do produto, se disponível>",
          "origem": "<Origem (ex.: nacional, importado)>",
          "categoria": ["Categoria hierarquica do produto - categoria genérica", "categoria intermediaria", "categoria especifica"]
        }
      },
      "tags": [
        "tag semantica 1",
        "tag semantica 2",
        "tag semantica 3",
        "tag semantica 4",
        "tag semantica 5"
      ]
    },
    "unidade_comercializacao": "<Unidade de comercialização>",
    "quantidade_comercializada": {
      "valor": "<Número>",
      "unidade": "<Unidade>"
    },
    "quantidade_convertida": {
      "valor": "<Número convertido>",
      "unidade": "<Unidade convertida>"
    },
    "valor_unitario": {
      "valor": "<Valor unitário>",
      "moeda": "<Código da moeda (ex.: BRL, USD)>"
    }
  }
}
```
'''
	prompt_template = "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n"
	example = """**Exemplo de saída:**

```json
{
  "produto": {
    "id_item": "1",
    "id_nfe": "1234567890",
    "ncm": "01012100",
    "gtin": "7891234567890",
    "descricao": {
      "original": "CAIXA C/ 12 GARRAFAS AGUA 500ML",
      "enriquecida": {
        "produto_base": "Água Mineral",
        "produto_detalhado": "Uma caixa com 12 garrafas de 500 ml de água mineral",
        "detalhes_extraidos": {
          "embalagem": {
            "tipo": "Caixa",
            "quantidade": {
              "valor": 12,
              "unidade": "unidades"
            }
          },
          "caracteristicas": {
            "dimensoes": null,
            "peso": null,
            "volume_por_unidade": {
              "valor": 500,
              "unidade": "mL"
            }
          }
        },
        "informacoes_adicionais": {
          "marca": null,
          "origem": "Nacional",
          "categoria": ["Bebidas", "água", "água mineral sem gas"]
        }
      },
      "tags": ["água mineral", "caixa de 12 garrafas de 500ml"]
    },
    "unidade_comercializacao": "UN",
    "quantidade_comercializada": {
      "valor": 24,
      "unidade": "unidades"
    },
    "quantidade_convertida": {
      "valor": 12,
      "unidade": "caixas"
    },
    "valor_unitario": {
      "valor": 2.50,
      "moeda": "BRL"
    }
  }
}
```"""
	@abstractmethod
	def generate(self, text: str) -> str:
		"""
		Generates a response based on the input text.
		Must be implemented by subclasses.
		"""
		pass

class NFeMinerLocalModel(NFeMinerBaseGenerateModel):
  """
    Local implementation of a language model, compatible with the BaseModel standard.

    This class loads a fine-tuned model (such as one produced by NFeFinetuner) and enables
    generating responses based on input text, formatting it with a standard Instruct-style prompt.

    Parameters:
      model_path (str): Path to the directory where the model and tokenizer are saved.

    Methods:
      generate(text: str) -> str:
      Generates a response based on the input text using the loaded model.
  """

	# def __init__(self, model_name:str="../models/llama3.1-nfe"):
	# 	from transformers import AutoTokenizer, AutoModelForCausalLM
	# 	import torch

	# 	self.tokenizer = AutoTokenizer.from_pretrained(model_name)
	# 	self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

	# def generate(self, text: str) -> str:
	# 	"""	
	# 	Generates a response for the provided text using the loaded model.

	# 	Args:
	# 			text (str): Input text to be used in the prompt.

	# 	Returns:
	# 			str: Response generated by the model, with special tokens removed.
	# 	"""

	# 	prompt = self.prompt_template.format(self.instruction, text)

	# 	inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
	# 	gen_idx = inputs['input_ids'].shape[1]

	# 	outputs = self.model.generate(**inputs, max_new_tokens=4096, use_cache=True)

	# 	response = self.tokenizer.batch_decode(outputs[:, gen_idx:], skip_special_tokens=True)[0]
	# 	return response.strip()

  def __init__(self, model_name: str = "../models/llama3.1-nfe"):
      from unsloth import FastLanguageModel
      from transformers import pipeline
      self.model, self.tokenizer = FastLanguageModel.from_pretrained(
          model_name = model_name,
          max_seq_length = 8192,        # ou conforme sua config
          dtype = None,                 # autodetecta fp16/bf16
          load_in_4bit = True,          # necessário para modelos quantizados
          device_map = "auto",
      )

      FastLanguageModel.for_inference(self.model) # Ativa modo de inferência (ativa flash-attn2, etc)

      self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto", do_sample=False, return_full_text=False)

  def generate(self, text: str) -> str:
      prompt = self.prompt_template.format(self.instruction, text)

      outputs = self.pipe(prompt, max_new_tokens=4096)
      return outputs[0]["generated_text"].strip()

class NFeMinerGPTModel(NFeMinerBaseGenerateModel):
	"""
	Implementation of the BaseModel interface using the OpenAI API (e.g., GPT-3.5, GPT-4).

	This class acts as a wrapper for communicating with models hosted by OpenAI,
	such as GPT-3.5-turbo or GPT-4, using the official API.

	Parameters:
			api_key (str): OpenAI API key.
			model_name (str): Name of the model to be used (default: "gpt-4").

	Methods:
			generate(text: str) -> str:
					Sends the input text as a prompt to the model and returns the generated response.
	"""

	def __init__(self, api_key: str, model_name="gpt-4"):
		import openai
		from openai import OpenAI

		openai.api_key = api_key
		self.model_name = model_name
		self.base_url = "https://openrouter.ai/api/v1"
		self.client = OpenAI(base_url=self.base_url, api_key=api_key)

	def generate(self, text: str) -> str:
		"""
		Generates a response based on the provided text using the OpenAI API.

		Args:
				text (str): Input text to be sent as a prompt.

		Returns:
				str: Textual response generated by the OpenAI model.
		"""

		prompt = self.prompt_template.format(f'{self.instruction}\n{self.example}', text)
		response = self.client.chat.completions.create(
				model=self.model_name,
				messages=[{"role": "user", "content": prompt}]
		)
		return response.choices[0].message.content

class NFeMinerOllamaModel(NFeMinerBaseGenerateModel):
	"""
	Implementation of the BaseModel interface using the OpenAI API (e.g., GPT-3.5, GPT-4).

	This class acts as a wrapper for communicating with models hosted by OpenAI,
	such as GPT-3.5-turbo or GPT-4, using the official API.

	Parameters:
			api_key (str): OpenAI API key.
			model_name (str): Name of the model to be used (default: "gpt-4").

	Methods:
			generate(text: str) -> str:
					Sends the input text as a prompt to the model and returns the generated response.
	"""

	def __init__(self, model_name):
		from langchain_ollama import OllamaLLM
		import subprocess
		subprocess.run(["ollama", "pull", model_name], check=True)
		self.model_name = model_name
		self.llm = OllamaLLM(model=model_name)


	def generate(self, text: str) -> str:
		"""
		Generates a response based on the provided text using the model.

		Args:
				text (str): Input text to be sent as a prompt.

		Returns:
				str: Textual response generated by the model.
		"""

		prompt = self.prompt_template.format(f'{self.instruction}\n{self.example}', text)
		return self.llm.invoke(prompt)

class NFeMinerVLLMModel(NFeMinerBaseGenerateModel):
    def __init__(self, model_name: str):
      import os, torch
      from transformers import AutoModelForCausalLM, AutoTokenizer
      from peft import PeftModel, PeftConfig
      from vllm import LLM, SamplingParams

      try:
        config = PeftConfig.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        merged_model = PeftModel.from_pretrained(base_model, model_name)
        merged_model = merged_model.merge_and_unload()
        load_path = os.path.join("cache", 'vllm', "merged")
        merged_model.save_pretrained(load_path)
        print(f"[INFO] Modelo mergeado salvo em {load_path}")
      except Exception as e:
        print(f"[INFO] Não é LoRA ou merge falhou: {e}")
        load_path = model_name

      self.model = LLM(model=load_path)
      self.sampling_params = SamplingParams(max_tokens=self.model.model_config.max_model_len)

    def generate(self, text: str) -> str:
      prompt = self.prompt_template.format(self.instruction, text)
      outputs = self.llm.generate(prompt, self.sampling_params)
      return outputs[0].outputs[0].text.strip()

__all__ = ["NFeMinerJSONValidator", "NFeMinerBaseGenerateModel", "NFeMinerLocalModel", "NFeMinerGPTModel", "NFeMinerOllamaModel"]
