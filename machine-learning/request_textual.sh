json_textual=$(cat request_textual.json)
echo "$json_textual"

curl -X POST "http://localhost:3003/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "entries=$json_textual" \
  -F "text=A photo of a cat"