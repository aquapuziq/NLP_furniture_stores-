from django.shortcuts import render
from .services.scraper import scrape_some_page_info
from .services.model_NLP import predict_products_from_text

def index(request):
    context = {
        "url": "",
        "products": [],
        "error": None,
        "submitted": False,
    }

    if request.method == "POST":
        context["submitted"] = True
        url = request.POST.get("url", "").strip()
        context["url"] = url

        try:
            text = scrape_some_page_info(url)
            products = predict_products_from_text(text)
            context["products"] = products

            print("URL:", url)
            print("PRODUCTS:", products)

        except Exception as e:
            context["error"] = str(e)
            print("ERROR:", e)

    return render(request, "extractor/index.html", context)