{{ site_header }}

## Notes

{% if notes %}
{% for note in notes %}
- [{{ note.title }}]({{ note.relative_url }}) *(updated {{ note.update_date }})*
{% endfor %}
{% else %}
\_^w^_/
{% endif %}
