{{ site_header }}

I'm Anna Allen; this is my website. My interests include functional programming, programming language design, machine learning, and computer systems/infrastructure. I encourage you to peruse my [blog](./blog), or [reach out to me](./contact) if you have questions or remarks, or a project that you feel my skills may be relevant to.

## Posts

{% for post in posts %}
*\[{{ post.start_date }}\]* [{{ post.title }}]({{ post.relative_url }})

{% endfor %}
