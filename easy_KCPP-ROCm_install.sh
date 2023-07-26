# #!/bin/bash

# git clone https://github.com/YellowRoseCx/koboldcpp-rocm.git -b main --depth 1 && \
# cd koboldcpp-rocm && \
# make LLAMA_HIPBLAS=1 -j4 && \
# ./koboldcpp.py

#!/bin/bash

if [ "$(basename "$PWD")" = "koboldcpp-rocm" ]; then
  echo "Already inside 'koboldcpp-rocm' directory."
else
  git clone https://github.com/YellowRoseCx/koboldcpp-rocm.git -b main --depth 1 && \
  cd "koboldcpp-rocm" || exit 1
fi


make LLAMA_HIPBLAS=1 -j4 && \
./koboldcpp.py
