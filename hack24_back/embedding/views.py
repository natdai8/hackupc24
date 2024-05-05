from django.shortcuts import render 

from rest_framework.response import Response
from rest_framework.views import APIView

from PIL import Image
from .utils.functions import initialize_chromadb,get_most_similar_url_images
class CalculateEmbeddingsAPIView(APIView):
    def post(self,request):

        image_file = request.FILES['image']
        img = Image.open(image_file)
        url_resnet,url_custom = get_most_similar_url_images(img)
        return Response({'urls-resnet': url_resnet,
                         'urls-custom': url_custom
                         })


    def get(self,request):
        initialize_chromadb()
        return Response({"message": "db init"})