local json = require 'cjson';

function fileExists(name)
  local f = io.open(name, "r")
  if f ~= nil then
    io.close(f)
    return true
  else
    return false
  end
end

function loadJson(name)
  f = io.open(name, 'r')
  data = f:read()
  return json.decode(data)
end
