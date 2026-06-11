{{ site_header }}

<style>
body {
  width: 70%;
  max-width: 90rem !important;
}

.photograph-gallery {
  display: grid;
  gap: 1rem;
}

.photograph-cluster {
  display: grid;
  gap: 0.9rem;
}

.photograph-cluster-landscape {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.photograph-cluster-portrait {
  grid-template-columns: repeat(5, minmax(0, 1fr));
}

.photograph-link {
  display: block;
  line-height: 0;
  background: none !important;
}

.photograph-link:hover,
.photograph-link:focus-visible {
  background: none !important;
}

.photograph-frame {
  margin: 0;
}

.photograph-image {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 4px;
}

@media (max-width: 1100px) {
  body {
    width: auto;
  }

  .photograph-cluster-landscape {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .photograph-cluster-portrait {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 600px) {
  .photograph-gallery {
    gap: 0.85rem;
  }

  .photograph-cluster {
    gap: 0.7rem;
  }

  .photograph-cluster-landscape {
    grid-template-columns: minmax(0, 1fr);
  }

  .photograph-cluster-portrait {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>

{% if page.photo_count %}
{{ page.gallery_markup }}
{% else %}
No photographs are published right now.
{% endif %}
