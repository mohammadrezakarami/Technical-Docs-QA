# Real QA Evaluation Report

## Summary

- Selected threshold: `0.0`
- Best threshold by QA F1: `0.0`
- Example count: `30`
- Answerable questions: `28`
- Explanatory questions: `7`
- Negative questions: `2`
- QA Exact Match: `0.3214`
- QA F1: `0.3959`
- Explanatory F1: `0.2263`
- Explanatory term coverage: `0.6857`
- Retrieval Hit@1: `0.7857`
- Retrieval Hit@3: `0.9286`
- Retrieval MRR: `0.8452`
- No-answer accuracy: `1.0000`
- Overall score: `0.4361`

## Threshold Sweep

| Threshold | QA EM | QA F1 | Expl. F1 | Expl. Coverage | Hit@1 | Hit@3 | MRR | No-answer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0.0` | `0.3214` | `0.3959` | `0.2263` | `0.6857` | `0.7857` | `0.9286` | `0.8452` | `1.0000` |

## Representative Gaps

### fastapi_headers_prefix
- Question: What prefix should custom proprietary headers use in FastAPI?
- Prediction: `None`
- Question type: `extractive`
- Exact Match: `0.0000`
- F1: `0.0000`
- Required term coverage: `0.0000`
- Hit@1: `1.0000`
- Top evidence chunk: `fastapi_6f626f41f8fa_chunk_024`
- Top source: `Response Headers - FastAPI`
- Top URL: https://fastapi.tiangolo.com/advanced/response-headers

### fastapi_streaming_response
- Question: Which response class should you declare to stream pure strings or binary data in FastAPI?
- Prediction: `default`
- Question type: `extractive`
- Exact Match: `0.0000`
- F1: `0.0000`
- Required term coverage: `0.0000`
- Hit@1: `1.0000`
- Top evidence chunk: `fastapi_471cea050831_chunk_000`
- Top source: `Stream Data - FastAPI`
- Top URL: https://fastapi.tiangolo.com/advanced/stream-data

### fastapi_stream_yield
- Question: What keyword can you use to send each chunk of data in turn with StreamingResponse in FastAPI?
- Prediction: `utf-8`
- Question type: `extractive`
- Exact Match: `0.0000`
- F1: `0.0000`
- Required term coverage: `0.0000`
- Hit@1: `1.0000`
- Top evidence chunk: `fastapi_471cea050831_chunk_004`
- Top source: `Stream Data - FastAPI`
- Top URL: https://fastapi.tiangolo.com/advanced/stream-data

### fastapi_install_command
- Question: What command installs FastAPI with the standard optional dependencies?
- Prediction: `pip`
- Question type: `extractive`
- Exact Match: `0.0000`
- F1: `0.5000`
- Required term coverage: `0.0000`
- Hit@1: `1.0000`
- Top evidence chunk: `fastapi_5503e14be81f_chunk_010`
- Top source: `Tutorial - User Guide - FastAPI`
- Top URL: https://fastapi.tiangolo.com/tutorial

### fastapi_additional_status_response
- Question: Which response class can you return directly to set additional status codes in FastAPI?
- Prediction: `Response`
- Question type: `extractive`
- Exact Match: `0.0000`
- F1: `0.0000`
- Required term coverage: `0.0000`
- Hit@1: `1.0000`
- Top evidence chunk: `fastapi_3d439ccba6c5_chunk_002`
- Top source: `Additional Status Codes - FastAPI`
- Top URL: https://fastapi.tiangolo.com/advanced/additional-status-codes
