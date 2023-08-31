ARG PYTHON_VERSION=3.10
FROM --platform=linux/amd64 public.ecr.aws/lambda/python:${PYTHON_VERSION}

WORKDIR /tmp
RUN pip install pip -U

COPY titiler /tmp/titiler
 # Echo the value of INSTALL_TYPE

ARG INSTALL_TYPE=prod
RUN echo "INSTALL_TYPE is set to: $INSTALL_TYPE"

RUN if [ "$INSTALL_TYPE" = "dev" ]; then \
      pip install /tmp/titiler/extensions['wms, cogeo, stac'] /tmp/titiler/application /tmp/titiler/core /tmp/titiler/mosaic "mangum>=0.10.0" -t /asset --no-binary pydantic ; \
    else \
      pip install "titiler.application==0.13.3" "mangum>=0.10.0" -t /asset --no-binary pydantic ; \
      pip install "titiler.application==0.14.0" "mangum>=0.10.0" -t /asset --no-binary pydantic
    fi

# Reduce package size and remove useless files
RUN cd /asset && find . -type f -name '*.pyc' | while read f; do n=$(echo $f | sed 's/__pycache__\///' | sed 's/.cpython-[0-9]*//'); cp $f $n; done;
RUN cd /asset && find . -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
#RUN cd /asset && find . -type f -a -name '*.py' -print0 | xargs -0 rm -f
RUN find /asset -type d -a -name 'tests' -print0 | xargs -0 rm -rf
RUN rm -rdf /asset/numpy/doc/ /asset/boto3* /asset/botocore* /asset/bin /asset/geos_license /asset/Misc

COPY lambda/handler.py /asset/handler.py

CMD ["echo", "hello world"]
