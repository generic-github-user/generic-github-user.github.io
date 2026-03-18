{{ site_header }}

# {{ post.title }}

<div class="post-meta post-meta-inline">
 started **{{ post.start_date }}** | updated **{{ post.update_date }}** | written in **{{ post.location }}** | {{ post.word_count }} words
</div>
<div class="post-meta post-meta-stacked">
  <span>started **{{ post.start_date }}**</span>
  <span>updated **{{ post.update_date }}**</span>
  <span>written in **{{ post.location }}**</span>
  <span>{{ post.word_count }} words</span>
</div>

{% if post.tags %}
tags: {% for tag in post.tags %}`{{ tag }}`{% if not loop.last %}, {% endif %}{% endfor %}
<br />
{% endif %}

{{ post.content }}
