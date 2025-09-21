from django.shortcuts import render
from .fuzzy_engine import run_fuzzy_diagnosis

def index(request):
    return render(request, 'diagnosis/index.html')

def result(request):
    if request.method == 'POST':
        data = {
            'academic_workload': float(request.POST['academic_workload']),
            'social_relationships': float(request.POST['social_relationships']),
            'average_sleep': float(request.POST['average_sleep']),
            'financial_concerns': float(request.POST['financial_concerns']),
            'academic_pressure': float(request.POST['academic_pressure']),
            'age': int(request.POST['age']),
        }
        output = run_fuzzy_diagnosis(data)
        return render(request, 'diagnosis/result.html', {'output': output})
    return render(request, 'diagnosis/index.html')
