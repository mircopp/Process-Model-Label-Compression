FROM ubuntu

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y graphviz libgraphviz-dev libopenblas-base libpng16-16 libxft2 swig wget python-dev \
          python-mock \
          python-pip \
          python2.7 \
          && apt-get clean
RUN pip install -U -r requirements.txt
RUN pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
RUN pip install 'ipython<6.0' protobuf numpy scipy jupyter syntaxnet-with-tensorflow

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]