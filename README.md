
```
pyenv install 3.8.13
```

```
pyenv versions
```

langchain 환경을 3.8.11 으로 생성
```
pyenv virtualenv 3.8.11 langchain
```

구성된 가상환경을 출력
```
pyenv virtualenvs
```

현재 경로를 `langchain` 환경으로 적용
```
pyenv local langchain
```

```
pip install chainlit langchain langchainhub gpt4all chromadb sentence-transformers GitPython
```

```
python build.py
```

```
chainlit run rag.py
```