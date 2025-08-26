from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List

class _Quantity(BaseModel):
    """
    Represents a value with an associated unit (e.g., volume, quantity).

    Attributes:
        valor (float | None): Numerical value.
        unidade (str | None): Unit of measurement (e.g., 'kg', 'mL').
    """
    valor: Optional[float] = None
    unidade: Optional[str] = None

class _Dimensional(BaseModel):
    """
    Represents dimensional attributes of a product (e.g., size).

    Attributes:
        valor (str | None): Dimensional value as a string.
        unidade (str | None): Measurement unit (e.g., 'cm', 'm').

    Validators:
        input_normalize: Ensures the 'valor' is converted to a string if it's a number.
    """
    valor: Optional[str] = None
    unidade: Optional[str] = None

    @model_validator(mode='before')
    def input_normalize(cls, data):
        """
        Converts numeric 'valor' into string form before validation.

        Args:
            data (dict): Input data.

        Returns:
            dict: Normalized data.
        """
        if isinstance(data, dict):
            value = data.get("valor")
            if value is not None:
                data['valor'] = str(value)
        return data

class _Weight(BaseModel):
    """
    Represents the weight of a product.

    Attributes:
        valor (float | None): Weight value.
        unidade (str | None): Weight unit (e.g., 'kg').
    """
    valor: Optional[float] = None
    unidade: Optional[str] = None

class _Characteristics(BaseModel):
    """
    Contains physical characteristics of the product.

    Attributes:
        dimensoes (_Dimensional): Dimensions.
        peso (_Weight): Weight.
        volume_por_unidade (_Quantity): Volume per unit.

    Validators:
        ensure_*: Ensures nested objects are initialized if None.
    """
    dimensoes: Optional[_Dimensional] = Field(default_factory=_Dimensional)
    peso: Optional[_Weight] = Field(default_factory=_Weight)
    volume_por_unidade: Optional[_Quantity] = Field(default_factory=_Quantity)

class _Packaging(BaseModel):
    """
    Packaging information of the product.

    Attributes:
        tipo (str | None): Type of packaging (e.g., box, bag).
        quantidade (_Quantity): Quantity of packages.

    Validators:
        ensure_Quantity: Initializes quantity if None.
    """
    tipo: Optional[str] = None
    quantidade: Optional[_Quantity] = Field(default_factory=_Quantity)

class _StructuredDetailsExtracted(BaseModel):
    """
    Structured details extracted from product description.

    Attributes:
        embalagem (_Packaging): Packaging information.
        caracteristicas (_Characteristics): Physical characteristics.

    Validators:
        ensure_structure: Initializes missing substructures.
    """
    embalagem: Optional[_Packaging] = Field(default_factory=_Packaging)
    caracteristicas: Optional[_Characteristics] = Field(default_factory=_Characteristics)

class _AdditionalMetadata(BaseModel):
    """
    Additional metadata about the product.

    Attributes:
        marca (str | None): Product brand.
        origem (str | None): Product origin (e.g., 'nacional').
        categoria (List[str]): Hierarchical category path.
    """
    marca: Optional[str] = None
    origem: Optional[str] = None
    categoria: Optional[List[str]] = Field(default_factory=list)

class _EnrichedDescription(BaseModel):
    """
    Enriched product description inferred from text.

    Attributes:
        produto_base (str | None): Base product name.
        produto_detalhado (str | None): Detailed inferred description.
        detalhes_extraidos (_StructuredDetailsExtracted): Structured attributes.
        informacoes_adicionais (_AdditionalMetadata): Additional metadata.
    """
    produto_base: Optional[str] = None
    produto_detalhado: Optional[str] = None
    detalhes_extraidos: Optional[_StructuredDetailsExtracted] = Field(default_factory=_StructuredDetailsExtracted)
    informacoes_adicionais: Optional[_AdditionalMetadata] = Field(default_factory=_AdditionalMetadata)

class _Description(BaseModel):
    """
    Original and enriched product description.

    Attributes:
        original (str): Raw text from the invoice.
        enriquecida (_EnrichedDescription): Enriched structured description.
        tags (List[str]): Semantic tags extracted or inferred.
    """
    original: str
    enriquecida: Optional[_EnrichedDescription] = Field(default_factory=_EnrichedDescription)
    tags: Optional[List[str]] = Field(default_factory=list)

class _UnitPrice(BaseModel):
    """
    Unit price of the product.

    Attributes:
        valor (float | None): Monetary value.
        moeda (str | None): Currency (default: 'BRL').
    """
    valor: Optional[float] = None
    moeda: Optional[str] = "BRL"

class _Product(BaseModel):
    """
    Main product schema, as expected in the JSON output.

    Attributes:
        id_item (int): Item identifier.
        id_nfe (str | None): Invoice ID.
        ncm (str | None): NCM classification code.
        gtin (str | None): GTIN barcode.
        descricao (_Description): Product description.
        unidade_comercializacao (str | None): Commercial unit.
        quantidade_comercializada (_Quantity): Quantity sold.
        valor_unitario (_UnitPrice): Unit price.
    """
    id_item: int
    id_nfe: Optional[str] = None
    ncm: Optional[str] = None
    gtin: Optional[str] = None
    descricao: _Description
    unidade_comercializacao: Optional[str] = None
    quantidade_comercializada: _Quantity = Field(default_factory=_Quantity)
    valor_unitario: _UnitPrice

class NFeMinerJSONValidator(BaseModel):
    """
    Top-level Pydantic schema used for validating enriched output.

    Attributes:
        produto (_Product): Full product structure.
    """
    produto: _Product

class NFeMinerBaseGenerateModel(ABC):
    """
    Abstract base class for models designed to extract structured data from electronic invoice (NFe) descriptions.

    This class defines the expected prompt structure, output format, and a standard method interface for generating
    enriched product data in JSON format. Subclasses must implement the `generate` method.

    Class Attributes:
        instruction (str): Instructions given to the model describing the expected behavior and output structure.
        prompt_template (str): Template for composing the full prompt to be sent to the model.
        input_template (str): Template for formatting invoice and item fields into a prompt for the model.
        example (str): An example of a valid model response for reference.

    Methods:
        generate(text: str) -> str:
            Abstract method. Should generate a raw response given a prompt.
        
        json_generate(invoice_id, item_id, ncm_code, gtin_code, sales_unit, quantity_sold, unit_price, description) -> dict:
            Generates enriched JSON data from basic product and invoice fields.
    """
    instruction = """Você é um modelo de linguagem treinado para processar dados de notas fiscais eletrônicas (NFEs) em português. Sua tarefa é analisar a descrição do item e outros campos fornecidos, extrair informações estruturadas e retornar um JSON formatado conforme uma estrutura predefinida.

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
"""

    prompt_template = "{}\n{}"

    input_template = "ID NFe: {invoice_id}\nID Item: {item_id}\nNomenclatura Comum do Mercosul (NCM): {ncm_code}\nGlobal Trade Item Number (GTIN): {gtin_code}\nUnidade de Comercialização: {sales_unit}\nQuantidade Comercializada: {quantity_sold}\nValor Unitário: {unit_price}\nDescrição: {description}"
    
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
        Generates a raw model response based on the given input text.

        This method must be implemented by any subclass using a specific model architecture
        (e.g., local model, OpenAI API, Ollama, etc.).

        Args:
            text (str): Input text (prompt) to send to the model.

        Returns:
            str: The generated text response from the model.
        """
        pass
  
    def json_generate(self, invoice_id: str, item_id: str, ncm_code: str, gtin_code: str, sales_unit: str, quantity_sold: float, unit_price: float, description: str) -> dict:
        """
        Generates a structured JSON representation of an invoice item using the language model.

        This method constructs a prompt using invoice and product details, invokes the model,
        and parses the response into a validated JSON structure.

        Args:
            invoice_id (str): Unique identifier of the electronic invoice (NFe).
            item_id (str): Identifier of the item within the invoice.
            ncm_code (str): Mercosur Common Nomenclature (NCM) code used for tax classification.
            gtin_code (str): Global Trade Item Number (GTIN), also known as the barcode.
            sales_unit (str): Unit of measurement used for selling the item (e.g., 'UN', 'kg').
            quantity_sold (float): Quantity of the item sold.
            unit_price (float): Unit price of the item.
            description (str): Raw item description from the invoice.

        Returns:
            dict: Validated structured JSON dictionary containing enriched product data.

        Raises:
            ValueError: If the model output cannot be parsed as valid JSON or have a incorrect format.
        """
        def json_strip(x):
            start = x.find('{')
            if start == -1:
                raise ValueError("Opening '{' not found in model response.")

            brace_count = 0
            for i in range(start, len(x)):
                if x[i] == '{':
                    brace_count += 1
                elif x[i] == '}':
                    brace_count -= 1

                if brace_count == 0:
                    x = x[start:i+1]
                    break

            if brace_count != 0:
                raise ValueError(f"Unmatched closing braces in model response. Remaining: {brace_count}")
            x = x.replace('None', 'null')
            return x

        from json5 import loads

        prompt = self.input_template.format(invoice_id=invoice_id, item_id=item_id, ncm_code=ncm_code, gtin_code=gtin_code, sales_unit=sales_unit, quantity_sold=quantity_sold, unit_price=unit_price, description=description)
        response = json_strip(self.generate(prompt))

        parsed = NFeMinerJSONValidator.model_validate(loads(response))
        return parsed.model_dump()

class NFeMinerLocalModel(NFeMinerBaseGenerateModel):
    """
    Local implementation of a fine-tuned language model for structured NFe item extraction.

    This class loads a model (typically fine-tuned using `NFeFinetuner`) and applies it to
    generate structured JSON outputs from textual descriptions of items in Brazilian electronic invoices (NFes).
    It uses a standard instruction and formatting template defined in the base class to create consistent prompts.

    Attributes:
        model (PreTrainedModel): The loaded fine-tuned language model.
        tokenizer (PreTrainedTokenizer): Tokenizer associated with the loaded model.
        pipe (transformers.Pipeline): Hugging Face pipeline for deterministic text generation.

    Args:
        model_name (str, optional): Path to the local model directory. Defaults to "models/gemma3_12b-pt_hf".

    Methods:
        generate(text: str) -> str:
            Generates structured output from a single input text using the model.
    """

    def __init__(self, model_name: str = "models/gemma3_12b-pt_hf"):
        """
        Initializes the local model, tokenizer, and text-generation pipeline.

        Loads a fine-tuned model using Unsloth's `FastLanguageModel`, sets it for inference,
        and creates a Hugging Face pipeline for text generation with deterministic behavior.

        Args:
            model_name (str, optional): Local path or identifier of the model. Defaults to "models/gemma3_12b-pt_hf".
        """
        from unsloth import FastLanguageModel
        from transformers import pipeline

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, dtype=None, device_map="auto")
        FastLanguageModel.for_inference(self.model)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto", do_sample=False, return_full_text=False)

    def generate(self, text: str) -> str:
        """
        Generates a structured JSON response based on the input text using the local model.

        The input is formatted with an instruction and passed through the generation pipeline
        to produce a completion suitable for NFe data extraction.

        Args:
            text (str): Input prompt representing an item description from an NFe.

        Returns:
            str: The generated JSON-formatted string response from the model.
        """
        prompt = self.prompt_template.format(self.instruction, text)
        outputs = self.pipe(prompt, max_new_tokens=4096)
        return outputs[0]["generated_text"].strip()

class NFeMinerOpenRouterModel(NFeMinerBaseGenerateModel):
    """
    Implementation of the NFeMiner interface using OpenRouter-hosted language models.

    This class wraps interactions with the OpenRouter-compatible OpenAI API client,
    enabling the generation of structured responses for electronic invoice (NFe) data enrichment
    using language models such as GPT-4.1 via the OpenRouter platform.

    Attributes:
        model_name (str): Identifier for the model to use (e.g., "openai/gpt-4.1").
        base_url (str): URL endpoint for the OpenRouter API.
        client (OpenAI): OpenAI client instance configured for OpenRouter.

    Args:
        api_key (str): OpenRouter API key for authentication.
        model_name (str, optional): Model identifier registered with OpenRouter. Defaults to "openai/gpt-4.1".

    Methods:
        generate(text: str) -> str:
            Generates a structured response using the selected model via the OpenRouter API.
    """

    def __init__(self, api_key: str, model_name="openai/gpt-4.1"):
        """
        Initializes the OpenRouter model client for response generation.

        Args:
            api_key (str): API key to authenticate with the OpenRouter service.
            model_name (str, optional): Name or ID of the model to use. Defaults to "openai/gpt-4.1".
        """
        import openai
        from openai import OpenAI

        openai.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    def generate(self, text: str) -> str:
        """
        Generates a response from the OpenRouter-hosted model based on the given input.

        The prompt is composed using the instruction and example templates defined in the base class,
        followed by the provided text, and submitted to the chat completion API.

        Args:
            text (str): Raw text input to be processed by the model.

        Returns:
            str: The response text generated by the language model.
        """
        prompt = self.prompt_template.format(f'{self.instruction}\n{self.example}', text)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

class NFeMinerOllamaModel(NFeMinerBaseGenerateModel):
    """
    Implementation of the NFeMiner interface using a locally hosted Ollama model.

    This class connects to an Ollama server to run a specified local language model for
    processing electronic invoice item descriptions and generating structured enrichment.

    Attributes:
        model_name (str): Name of the model to be used via Ollama (must be pulled beforehand or automatically).
        client (ollama.Client): Client used to communicate with the Ollama server.
        options (dict): Generation parameters passed to the model (e.g., temperature, top_k, etc.).

    Args:
        model_name (str): Name of the Ollama model to use (e.g., "llama3", "mistral-nfe").
        host (str, optional): URL of the Ollama server. Defaults to "http://localhost:11434".
        options (dict, optional): Generation options for model behavior. Includes temperature, top_k, etc.
                                  Defaults to reasonable values for deterministic NFe inference.
    
    Methods:
        generate(text: str) -> str:
            Sends a prompt to the Ollama model and returns the generated response.
    """

    def __init__(self, model_name, host='http://localhost:11434', options={'temperature': 1, 'top_k': 64, 'top_p': 0.95, 'repeat_penalty': 1.1, 'seed': 777, 'num_predict': 16*1024}):
        """
        Initializes the Ollama model and pulls it from the registry if needed.

        Args:
            model_name (str): Name of the model to pull and use.
            host (str, optional): Address of the Ollama API server. Defaults to "http://localhost:11434".
            options (dict, optional): Generation options passed to the Ollama model.
        """
        from ollama import Client
        import subprocess

        subprocess.run(["ollama", "pull", model_name], check=True)
        self.model_name = model_name
        self.client = Client(host=host)
        self.options = options

    def generate(self, text: str) -> str:
        """
        Generates a response from the Ollama-hosted local model based on the provided input.

        Constructs a formatted prompt using instruction and example templates, then sends it
        to the Ollama model with the specified generation options.

        Args:
            text (str): The input text to be processed by the model.

        Returns:
            str: The response generated by the Ollama model.
        """
        prompt = self.prompt_template.format(f'{self.instruction}\n{self.example}', text)
        return self.client.generate(model=self.model_name, prompt=prompt, options=self.options)['response']

__all__ = ["NFeMinerJSONValidator", "NFeMinerBaseGenerateModel", "NFeMinerLocalModel", "NFeMinerGPTModel", "NFeMinerOllamaModel"]