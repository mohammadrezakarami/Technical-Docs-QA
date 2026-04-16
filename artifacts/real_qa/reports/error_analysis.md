# Error Analysis

## Counts

- Retrieval misses: `6`
- Reader/span issues: `15`
- False positives: `0`

## retrieval_miss

### fastapi_created_status_code
- Question: What status code does the FastAPI upsert example return when it creates a new item?
- Prediction: `OK" 200`
- Hit@1: `0.0000`
- F1: `0.0000`
- Top URL: https://fastapi.tiangolo.com/advanced/response-change-status-code

### pandas_read_csv
- Question: What function reads a CSV file in pandas?
- Prediction: `read_csv()`
- Hit@1: `0.0000`
- F1: `1.0000`
- Top URL: https://pandas.pydata.org/docs/user_guide/io.html

### fastapi_default_json_responses
- Question: By default, what kind of responses does FastAPI return?
- Prediction: `default values`
- Hit@1: `0.0000`
- F1: `0.0000`
- Top URL: https://fastapi.tiangolo.com/reference/fastapi

### fastapi_underlying_response_library
- Question: Most of the available FastAPI responses come directly from which library?
- Prediction: `Starlette`
- Hit@1: `0.0000`
- F1: `1.0000`
- Top URL: https://fastapi.tiangolo.com/advanced/templates

### pandas_groupby_method
- Question: Which pandas method returns a GroupBy object for grouping rows?
- Prediction: `DataFrameGroupBy`
- Hit@1: `0.0000`
- F1: `0.0000`
- Top URL: https://pandas.pydata.org/docs/reference/groupby.html

### pandas_read_parquet
- Question: What function reads a Parquet file in pandas?
- Prediction: `to_parquet`
- Hit@1: `0.0000`
- F1: `0.0000`
- Top URL: https://pandas.pydata.org/docs/reference/io.html

## reader_or_span_issue

### fastapi_headers_prefix
- Question: What prefix should custom proprietary headers use in FastAPI?
- Prediction: `None`
- Hit@1: `1.0000`
- F1: `0.0000`
- Top URL: https://fastapi.tiangolo.com/advanced/response-headers

### fastapi_streaming_response
- Question: Which response class should you declare to stream pure strings or binary data in FastAPI?
- Prediction: `default`
- Hit@1: `1.0000`
- F1: `0.0000`
- Top URL: https://fastapi.tiangolo.com/advanced/stream-data

### fastapi_stream_yield
- Question: What keyword can you use to send each chunk of data in turn with StreamingResponse in FastAPI?
- Prediction: `utf-8`
- Hit@1: `1.0000`
- F1: `0.0000`
- Top URL: https://fastapi.tiangolo.com/advanced/stream-data

### fastapi_install_command
- Question: What command installs FastAPI with the standard optional dependencies?
- Prediction: `pip`
- Hit@1: `1.0000`
- F1: `0.5000`
- Top URL: https://fastapi.tiangolo.com/tutorial

### fastapi_additional_status_response
- Question: Which response class can you return directly to set additional status codes in FastAPI?
- Prediction: `Response`
- Hit@1: `1.0000`
- F1: `0.0000`
- Top URL: https://fastapi.tiangolo.com/advanced/additional-status-codes

### pandas_to_csv
- Question: What method writes a DataFrame to a CSV file in pandas?
- Prediction: `to_excel()`
- Hit@1: `1.0000`
- F1: `0.0000`
- Top URL: https://pandas.pydata.org/docs/user_guide/io.html

### pandas_loc_iloc_recommendation
- Question: Which pandas access methods are recommended for production code when selecting and setting data?
- Prediction: `While standard Python / NumPy expressions for selecting and setting are intuitive and come in handy for interactive work, for production code, we recommend the optimized pandas data access methods, DataFrame.at() , DataFrame.iat() , DataFrame.loc() For production code, we recommended that you take advantage of the optimized pandas data access methods exposed in this chapter. See the MultiIndex / Advanced Indexing for However, since the type of the data to be accessed isn’t known in advance, directly using standard operators has some optimization limits.`
- Hit@1: `1.0000`
- F1: `0.1000`
- Top URL: https://pandas.pydata.org/docs/user_guide/10min.html

### fastapi_stream_pure_strings
- Question: Which FastAPI response class is used to stream pure strings or binary data?
- Prediction: `Info`
- Hit@1: `1.0000`
- F1: `0.0000`
- Top URL: https://fastapi.tiangolo.com/advanced/stream-data

### fastapi_jsonresponse_default_wrapper
- Question: Which FastAPI response class wraps returned content by default when returning JSON?
- Prediction: `Custom Response`
- Hit@1: `1.0000`
- F1: `0.0000`
- Top URL: https://fastapi.tiangolo.com/advanced/custom-response

### fastapi_response_headers_explained
- Question: How do you set custom response headers in FastAPI, and why does using a Response parameter work?
- Prediction: `Based on the documentation, and as the Response can be used frequently to set headers and cookies, FastAPI also provides it at fastapi.Response . Custom Headers You can also declare the Response parameter in dependencies, and set headers (and cookies) in them. Return a Response directly Response Headers Use a Response parameter You can declare a parameter of type Response in your path operation function`
- Hit@1: `1.0000`
- F1: `0.3333`
- Top URL: https://fastapi.tiangolo.com/advanced/response-headers

## false_positive

- None
