--[[
Shorten bare hyperlinks by dropping the protocol and trailing slash,
while keeping the actual href untouched.

Example: <https://example.com/> renders as example.com
]]

local function shorten(url)
  local cleaned = url:gsub('^https?://', '')
  cleaned = cleaned:gsub('/$', '')
  return cleaned
end

function Link(el)
  -- Only touch auto-linked URLs whose text matches the href
  local display = pandoc.utils.stringify(el.content)
  if display ~= el.target then
    return nil
  end

  if not el.target:match('^https?://') then
    return nil
  end

  local shortened = shorten(el.target)
  if shortened == display then
    return nil
  end

  el.content = { pandoc.Str(shortened) }
  return el
end
