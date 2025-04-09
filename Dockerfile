FROM public.ecr.aws/lambda/python:3.9
COPY requirements.txt .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
COPY model_download.py ${LAMBDA_TASK_ROOT}
RUN python ${LAMBDA_TASK_ROOT}/model_download.py

ENV HF_MODEL_DIR=/var/rsc
COPY handler.py ${LAMBDA_TASK_ROOT}
CMD ["handler.lambda_handler"]