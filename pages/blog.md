{{ site_header }}

## Blog

{% if posts %}
{% for post in posts %}
- [{{ post.title }}]({{ post.relative_url }}) *(updated {{ post.update_date }})*
{% endfor %}
{% else %}
No posts are published right now. Check back soon!
{% endif %}
