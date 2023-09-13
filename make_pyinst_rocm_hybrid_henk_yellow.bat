cd /d "%~dp0"
copy "C:\Program Files\AMD\ROCm\5.5\bin\hipblas.dll" .\ /Y
copy "C:\Program Files\AMD\ROCm\5.5\bin\rocblas.dll" .\ /Y
xcopy /E /I "C:\Program Files\AMD\ROCm\5.5\bin\rocblas" .\rocblas\
 
PyInstaller --noconfirm --onefile --collect-all customtkinter --clean --console --icon ".\niko.ico" --add-data "./klite.embd;." --add-data "./koboldcpp_default.dll;." --add-data "./koboldcpp_openblas.dll;." --add-data "./koboldcpp_failsafe.dll;." --add-data "./koboldcpp_noavx2.dll;." --add-data "./libopenblas.dll;." --add-data "./koboldcpp_clblast.dll;." --add-data "./clblast.dll;." --add-data "./koboldcpp_cublas.dll;." --add-data "./hipblas.dll;." --add-data "./rocblas.dll;." --add-data "./rwkv_vocab.embd;." --add-data "./rocblas;." --add-data "C:/Windows/System32/msvcp140.dll;." --add-data "C:/Windows/System32/vcruntime140_1.dll;." "./koboldcpp.py" -n "koboldcppRocm.exe"