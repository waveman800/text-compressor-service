import sys
import os
import re
import aiohttp
import time
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uvicorn
import requests
import json

from core.dynamic_compressor import DynamicContextCompressor
from core.tokenizer_wrapper import TokenizerWrapper
from core.model_config import (
    get_model_config, 
    get_all_models, 
    calculate_session_len,
    get_provider_for_model,
    get_available_models
)

app = FastAPI(title="Text Compression Service", description="Dynamic text compression service with OpenAI protocol support")

tokenizer = TokenizerWrapper()
compressor = DynamicContextCompressor(session_len=4096, tokenizer=tokenizer)


def clean_model_response(response_text: str) -> str:
    """清理模型返回的响应"""
    try:
        if response_text is None:
            return ""
        
        # 处理多重编码问题
        if isinstance(response_text, bytes):
            response_text = response_text.decode('utf-8', 'replace')
        
        # 确保是字符串类型
        response_text = str(response_text)
        
        # 处理Unicode转义字符（更安全的方式）
        try:
            # 只处理明确的Unicode转义序列，不改变其他内容
            response_text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), response_text)
        except Exception:
            pass
        
        # 移除多余的空格和换行
        response_text = re.sub(r'\s+', ' ', response_text).strip()
        
        return response_text
    except Exception as e:
        print(f"Error cleaning response: {e}")
        return response_text if response_text is not None else ""


def calculate_generated_tokens(response_text: str) -> int:
    """计算生成的token数"""
    try:
        return tokenizer.count_tokens(response_text)
    except Exception as e:
        print(f"Error calculating tokens: {e}")
        return 0


class CompressTextRequest(BaseModel):
    text: str = Field(..., description="要压缩的文本")
    current_prompt: str = Field(default="", description="当前的提示词")
    max_new_tokens: int = Field(default=256, description="生成新token的最大数量")
    session_len: int = Field(default=None, description="会话窗口的最大token数（不指定则根据model自动计算）")
    model_name: str = Field(default="gpt-3.5-turbo", description="模型名称，用于自动计算session_len")
    use_fast_compression: bool = Field(default=False, description="是否使用快速压缩模式")
    safety_margin: float = Field(default=0.1, description="安全余量（0-1），默认10%")

class CompressChatRequest(BaseModel):
    chat_history: List[Dict[str, Any]] = Field(..., description="聊天历史")
    current_prompt: str = Field(default="", description="当前的提示词")
    max_new_tokens: int = Field(default=256, description="生成新token的最大数量")
    session_len: int = Field(default=None, description="会话窗口的最大token数（不指定则根据model自动计算）")
    model_name: str = Field(default="gpt-3.5-turbo", description="模型名称，用于自动计算session_len")
    safety_margin: float = Field(default=0.1, description="安全余量（0-1），默认10%")

class CompressResponse(BaseModel):
    compressed_text: Optional[str] = Field(None, description="压缩后的文本")
    compressed_chat: Optional[List[Dict[str, Any]]] = Field(None, description="压缩后的聊天历史")
    was_compressed: bool = Field(..., description="是否进行了压缩")
    original_length: int = Field(..., description="原始token长度")
    compressed_length: int = Field(..., description="压缩后的token长度")

class OpenAIMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class OpenAICompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 256
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "text-compression-service"}


@app.get("/models")
async def list_models():
    models = get_available_models()
    model_list = []
    for model_id, config in models.items():
        model_list.append({
            "id": model_id,
            "name": config.name,
            "context_length": config.context_length,
            "max_output_tokens": config.max_output_tokens,
            "description": config.description,
            "provider": config.provider
        })
    return {"models": model_list}


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    config = get_model_config(model_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    provider = get_provider_for_model(model_id)
    
    return {
        "id": model_id,
        "name": config.name,
        "context_length": config.context_length,
        "max_output_tokens": config.max_output_tokens,
        "description": config.description,
        "provider": config.provider,
        "provider_name": provider.provider_name if provider else None,
        "provider_api_base": provider.api_base_url if provider else None
    }


@app.get("/providers")
async def list_providers():
    from core.model_config import get_llm_providers
    providers = get_llm_providers()
    provider_list = []
    for provider_id, config in providers.items():
        provider_list.append({
            "id": provider_id,
            "name": config.provider_name,
            "api_base_url": config.api_base_url,
            "has_api_key": bool(config.api_key)
        })
    return {"providers": provider_list}


@app.post("/compress/text", response_model=CompressResponse)
async def compress_text(request: CompressTextRequest):
    try:
        if request.session_len is None:
            session_len = calculate_session_len(request.model_name, request.safety_margin)
            if session_len is None:
                raise HTTPException(status_code=400, detail=f"Unknown model: {request.model_name}")
        else:
            session_len = request.session_len
        
        compressor.session_len = session_len
        compressor.use_fast_compression = request.use_fast_compression
        
        original_length = tokenizer.count_tokens(request.text)
        
        compressed_text, was_compressed = compressor.dynamic_compress(
            text=request.text,
            current_prompt=request.current_prompt,
            max_new_tokens=request.max_new_tokens
        )
        
        compressed_length = tokenizer.count_tokens(compressed_text)
        
        return CompressResponse(
            compressed_text=compressed_text,
            was_compressed=was_compressed,
            original_length=original_length,
            compressed_length=compressed_length
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")


@app.post("/compress/chat", response_model=CompressResponse)
async def compress_chat(request: CompressChatRequest):
    try:
        if request.session_len is None:
            session_len = calculate_session_len(request.model_name, request.safety_margin)
            if session_len is None:
                raise HTTPException(status_code=400, detail=f"Unknown model: {request.model_name}")
        else:
            session_len = request.session_len
        
        compressor.session_len = session_len
        
        original_length = tokenizer.count_tokens(request.chat_history)
        
        compressed_chat, was_compressed = compressor.compress_chat_history(
            chat_history=request.chat_history,
            current_prompt=request.current_prompt,
            max_new_tokens=request.max_new_tokens
        )
        
        compressed_length = tokenizer.count_tokens(compressed_chat)
        
        return CompressResponse(
            compressed_chat=compressed_chat,
            was_compressed=was_compressed,
            original_length=original_length,
            compressed_length=compressed_length
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat compression failed: {str(e)}")


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAICompletionRequest):
    try:
        model_config = get_model_config(request.model)
        if model_config is None:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

        session_len = calculate_session_len(request.model, safety_margin=0.1)
        if session_len is None:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

        provider = get_provider_for_model(request.model)
        print(f"[DEBUG] Request model: {request.model}")
        print(f"[DEBUG] Model config provider: {model_config.provider}")
        print(f"[DEBUG] Obtained provider: {provider}")
        if provider is None:
            raise HTTPException(status_code=400, detail=f"No provider configured for model: {request.model}")

        compressor.session_len = session_len

        max_new_tokens = request.max_tokens if request.max_tokens else model_config.max_output_tokens + 800
        compressor.max_new_tokens = max_new_tokens

        chat_history = [msg.dict() for msg in request.messages]

        current_prompt = ""
        for msg in reversed(chat_history):
            if msg["role"] == "user":
                current_prompt = msg["content"]
                break

        print(f"[DEBUG] Original chat_history length: {len(chat_history)}")
        print(f"[DEBUG] Original chat_history: {chat_history}")
        print(f"[DEBUG] Current prompt length: {len(current_prompt)}")
        print(f"[DEBUG] Compressing chat history...")
        compressed_history, was_compressed = compressor.compress_chat_history(
            chat_history=chat_history,
            current_prompt=current_prompt,
            max_new_tokens=max_new_tokens
        )
        print(f"[DEBUG] Compression result: was_compressed={was_compressed}")
        print(f"[DEBUG] Compressed chat_history length: {len(compressed_history)}")
        print(f"[DEBUG] Compressed chat_history: {compressed_history}")
        
        # 计算输入的token数
        input_token_count = 0
        for msg in compressed_history:
            if 'content' in msg and msg['content']:
                input_token_count += tokenizer.count_tokens(msg['content'])
        print(f"[DEBUG] Input token count: {input_token_count}")

        proxy_request = request.dict()
        proxy_request["messages"] = compressed_history
        # 确保将计算的max_new_tokens传递给模型服务
        proxy_request["max_tokens"] = max_new_tokens

        headers = {
            "Content-Type": "application/json"
        }

        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

        api_url = f"{provider.api_base_url}/chat/completions"
        print(f"[DEBUG] Proxying request to: {api_url}")
        print(f"[DEBUG] Request payload size: {len(json.dumps(proxy_request)):,} bytes")
        print(f"[DEBUG] Number of messages: {len(proxy_request['messages'])}")
        print(f"[DEBUG] Request stream parameter: {request.stream}")
        print(f"[DEBUG] Proxy request stream parameter: {proxy_request.get('stream')}")

        try:
            if request.stream:
                # 使用requests库处理流式响应
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=proxy_request,
                    stream=True
                )
            else:
                # 处理非流式响应
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=proxy_request,
                    stream=False
                )
            
            # 检查响应状态
            if response.status_code != 200:
                # 尝试获取错误信息
                error_detail = ""
                try:
                    error_detail = response.text
                except:
                    pass
                raise HTTPException(status_code=response.status_code, detail=f"Model service error: {error_detail}")
            
            # 无论请求是否为流式，都检查响应是否为流式格式
            is_stream_response = False
            try:
                # 尝试读取第一行来判断是否为流式响应
                if response.headers.get('content-type', '').startswith('text/event-stream'):
                    is_stream_response = True
            except:
                pass
            
            # 处理流式响应
            if request.stream:
                def stream_response():
                    try:
                        print("[DEBUG] Starting stream response")
                        for i, line in enumerate(response.iter_lines()):
                            print(f"[DEBUG] Received line {i}: length={len(line)} bytes")
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith('data: '):
                                    data = line[6:].strip()
                                    if data == '[DONE]':
                                        print(f"[DEBUG] Received DONE signal")
                                        yield 'data: [DONE]\n\n'
                                        break
                                    try:
                                        json_data = json.loads(data)
                                        print(f"[DEBUG] JSON parsed successfully: has_choices={len(json_data.get('choices', []))}")
                                        if 'choices' in json_data and len(json_data['choices']) > 0:
                                            choice = json_data['choices'][0]
                                            if 'delta' in choice:
                                                delta = choice['delta']
                                                if 'content' in delta:
                                                    if delta['content'] is not None:
                                                        content_len = len(delta['content'])
                                                        print(f"[DEBUG] Processing content: length={content_len}, first_100_chars={delta['content'][:100] if content_len > 100 else delta['content']}")
                                                        # 清理响应内容 - 不截断文本
                                                        cleaned_content = clean_model_response(delta['content'])
                                                        delta['content'] = cleaned_content
                                                    else:
                                                        print(f"[DEBUG] Content is None, skipping cleaning")
                                                json_data['choices'][0]['delta'] = delta
                                        response_line = f'data: {json.dumps(json_data)}\n\n'
                                        print(f"[DEBUG] Sending response line: length={len(response_line)}")
                                        yield response_line
                                    except json.JSONDecodeError as e:
                                        print(f"[DEBUG] JSON decode error: {e}, data: {data}")
                                        # 尝试修复可能的JSON格式问题
                                        try:
                                            # 移除可能的控制字符
                                            cleaned_data = re.sub(r'[\x00-\x1f]', '', data)
                                            print(f"[DEBUG] Removed control characters: original_length={len(data)}, cleaned_length={len(cleaned_data)}")
                                            # 尝试再次解析
                                            json_data = json.loads(cleaned_data)
                                            response_line = f'data: {json.dumps(json_data)}\n\n'
                                            print(f"[DEBUG] Fixed JSON successfully: sending response length={len(response_line)}")
                                            yield response_line
                                        except json.JSONDecodeError:
                                            # 如果仍然解析失败，记录错误但不中断流
                                            print(f"[DEBUG] Failed to fix JSON: {data}")
                                            # 尝试将原始内容作为文本返回
                                            error_data = {"choices": [{"delta": {"content": data}}]}
                                            response_line = f'data: {json.dumps(error_data)}\n\n'
                                            print(f"[DEBUG] Sending fallback response: length={len(response_line)}")
                                            yield response_line
                    except requests.exceptions.RequestException as e:
                        print(f"Stream response error: {e}")
                        yield f'data: {{"error": {{"message": "Stream connection closed", "type": "connection_error"}}}}\n\n'
                
                return StreamingResponse(stream_response(), media_type="text/event-stream")
            else:
                # 处理非流式响应请求（即使上游返回流式数据）
                print("[DEBUG] Processing non-stream response request")
                print(f"[DEBUG] Response content-type: {response.headers.get('content-type')}")
                print(f"[DEBUG] Is stream response: {is_stream_response}")
                try:
                    full_response_data = None
                    full_content = ""
                    
                    # 如果是流式响应格式，需要聚合所有chunk
                    if is_stream_response:
                        print("[DEBUG] Detected stream response format, aggregating chunks...")
                        
                        # 确保响应是流式的，重新创建响应对象以支持迭代
                        response = requests.post(
                            api_url,
                            headers=headers,
                            json=proxy_request,
                            stream=True
                        )
                        
                        for i, line in enumerate(response.iter_lines()):
                            if line:
                                line = line.decode('utf-8')
                                print(f"[DEBUG] Processing line {i}: {line}")
                                if line.startswith('data: '):
                                    data = line[6:].strip()
                                    if data == '[DONE]':
                                        break
                                    try:
                                        json_data = json.loads(data)
                                        full_response_data = json_data
                                        # 拼接所有chunk的content
                                        if 'choices' in json_data and json_data['choices']:
                                            choice = json_data['choices'][0]
                                            if 'delta' in choice and 'content' in choice['delta']:
                                                if choice['delta']['content']:
                                                    full_content += choice['delta']['content']
                                    except json.JSONDecodeError as je:
                                        print(f"[DEBUG] JSON decode error in line {i}: {je}")
                                        print(f"[DEBUG] Problematic data: {data}")
                                        pass
                    else:
                        # 非流式响应，直接解析
                        print("[DEBUG] Detected non-stream response format, parsing directly...")
                        response_content = response.text
                        print(f"[DEBUG] Raw response content length: {len(response_content)}")
                        print(f"[DEBUG] Raw response content: {response_content}")
                        
                        try:
                            full_response_data = json.loads(response_content)
                        except json.JSONDecodeError as je:
                            print(f"[DEBUG] JSON decode error in non-stream response: {je}")
                            # 尝试检查是否是SSE格式
                            if response_content.strip().startswith('data: '):
                                print("[DEBUG] Response appears to be SSE format despite content-type")
                                is_stream_response = True
                                
                                # 重新创建流式响应对象
                                response = requests.post(
                                    api_url,
                                    headers=headers,
                                    json=proxy_request,
                                    stream=True
                                )
                                
                                # 聚合流式响应
                                for i, line in enumerate(response.iter_lines()):
                                    if line:
                                        line = line.decode('utf-8')
                                        print(f"[DEBUG] Processing line {i}: {line}")
                                        if line.startswith('data: '):
                                            data = line[6:].strip()
                                            if data == '[DONE]':
                                                break
                                            try:
                                                json_data = json.loads(data)
                                                full_response_data = json_data
                                                # 拼接所有chunk的content
                                                if 'choices' in json_data and json_data['choices']:
                                                    choice = json_data['choices'][0]
                                                    if 'delta' in choice and 'content' in choice['delta']:
                                                        if choice['delta']['content']:
                                                            full_content += choice['delta']['content']
                                            except json.JSONDecodeError as je2:
                                                print(f"[DEBUG] JSON decode error in line {i}: {je2}")
                                                pass
                            else:
                                raise HTTPException(status_code=500, detail="Failed to parse model response")
                    
                    print(f"[DEBUG] Aggregated content: {full_content}")
                    print(f"[DEBUG] Full response data: {full_response_data}")
                    
                    if not full_response_data:
                        raise HTTPException(status_code=500, detail="No valid response received from model")
                    
                    # 构建完整的非流式响应结构
                    print("[DEBUG] Building complete non-stream response structure")
                    
                    # 如果有聚合的内容，确保响应结构完整
                    if full_content:
                        # 创建完整的响应结构
                        complete_response = {
                            "id": full_response_data.get("id", "chatcmpl-123"),
                            "object": "chat.completion",
                            "created": full_response_data.get("created", int(time.time())),
                            "model": full_response_data.get("model", request.model),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": clean_model_response(full_content)
                                    },
                                    "logprobs": None,
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": full_response_data.get("usage", None)
                        }
                        
                        print(f"[DEBUG] Complete response structure created")
                        print(f"[DEBUG] Response content length: {len(complete_response['choices'][0]['message']['content'])}")
                        
                        return JSONResponse(content=complete_response)
                    elif full_response_data:
                        # 清理现有响应内容
                        if "choices" in full_response_data and full_response_data["choices"]:
                            choice = full_response_data["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                print("[DEBUG] Using existing message content")
                                choice["message"]["content"] = clean_model_response(choice["message"]["content"])
                            elif "delta" in choice and "content" in choice["delta"]:
                                print("[DEBUG] Using delta content for message")
                                # 如果只有最后一个chunk的content，使用它构造message字段
                                message_content = choice["delta"]["content"] or ""
                                choice["message"] = {
                                    "role": "assistant",
                                    "content": clean_model_response(message_content)
                                }
                                del choice["delta"]
                        
                        print("[DEBUG] Returning final JSON response")
                        return JSONResponse(content=full_response_data)
                    else:
                        raise HTTPException(status_code=500, detail="No valid response content received from model")
                except Exception as e:
                    print(f"[DEBUG] Non-stream processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    raise HTTPException(status_code=500, detail=f"Failed to process non-stream response: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to model service: {str(e)}")
        
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] Request exception: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to model service: {str(e)}")
    except Exception as e:
        print(f"[DEBUG] Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Proxy request failed: {str(e)}")


@app.get("/v1/models")
async def openai_models():
    try:
        models = get_available_models()
        model_list = []
        for model_id, config in models.items():
            model_list.append({
                "id": model_id,
                "object": "model",
                "created": 1677610602,
                "owned_by": config.provider or "custom"
            })
        
        return {
            "object": "list",
            "data": model_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def openai_proxy(request: Request, path: str):
    try:
        if path.startswith("chat/completions"):
            return await openai_chat_completions(await request.json())
        
        if path == "models":
            return await openai_models()
        
        raise HTTPException(status_code=404, detail=f"Endpoint not found: /v1/{path}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy request failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
