{{ site_header }}

Redirecting you to something random on the site. If you are not redirected automatically, <a id="random-fallback-link" href="/index">pick a page manually</a>.

<script>
document.addEventListener('DOMContentLoaded', () => {
  const targets = {{ random_targets_json | safe }};
  const fallbackLink = document.getElementById('random-fallback-link');
  if (!Array.isArray(targets) || targets.length === 0) {
    if (fallbackLink) {
      fallbackLink.textContent = 'return home';
      fallbackLink.href = '/index';
    }
    return;
  }
  const choice = targets[Math.floor(Math.random() * targets.length)];
  if (fallbackLink && choice) {
    fallbackLink.textContent = choice.title || choice.url;
    fallbackLink.href = choice.url;
  }
  window.location.replace(choice.url);
});
</script>

<noscript>
JavaScript is required to select a random destination automatically. Here are a few options you can choose manually:

<ul>
{% for entry in site_pages %}
  <li><a href="{{ entry.url }}">{{ entry.title }}</a></li>
{% endfor %}
{% for post in posts %}
  <li><a href="{{ post.url }}">{{ post.title or post.name }}</a></li>
{% endfor %}
{% for note in notes %}
  <li><a href="{{ note.url }}">{{ note.title or note.name }}</a></li>
{% endfor %}
</ul>
</noscript>
