json_visual=$(cat request_visual.json)
echo "$json_visual"

curl -X POST "http://localhost:3003/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "entries=$json_visual" \
  -F "image=@images/ssd_dog.jpg"
