{{ site_header }}

# {{ post.title }}

 started **{{ post.start_date }}** | updated **{{ post.update_date }}** | written in **{{ post.location }}** | {{ post.word_count }} words

{% if post.tags %}
tags: {% for tag in post.tags %}`{{ tag }}`{% if not loop.last %}, {% endif %}{% endfor %}
{% endif %}

{{ post.content }}
