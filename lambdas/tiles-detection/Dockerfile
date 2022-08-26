FROM public.ecr.aws/lambda/python:3.8

# for opencv
RUN yum install mesa-libGL -y

WORKDIR /gow-be
# Copy function code
COPY tiles-detection.py ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY ./requirements.txt  .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "tiles-detection.handler" ]
