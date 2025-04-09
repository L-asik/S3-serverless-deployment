FROM public.ecr.aws/lambda/python:3.9
COPY requirements.txt .
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
COPY pretrained_download.py ${LAMBDA_TASK_ROOT}
RUN python ${LAMBDA_TASK_ROOT}/pretrained_download.py
COPY handler.py ${LAMBDA_TASK_ROOT}
CMD ["handler.lambda_handler"]