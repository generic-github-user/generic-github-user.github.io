{{ site_header }}

# {{ post.title }}

started **{{ post.start_date }}** | updated **{{ post.update_date }}** | written in **{{ post.location }}**

{% if post.history_url %}
<div class="history-link-inline">
  <a href="{{ post.history_url }}">View revision history</a>
</div>
{% endif %}

{{ post.content }}
