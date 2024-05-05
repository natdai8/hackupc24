from django.urls import path

from .views import CalculateEmbeddingsAPIView
urlpatterns = [
    path('fashion',CalculateEmbeddingsAPIView.as_view(),name='embeddings')
]