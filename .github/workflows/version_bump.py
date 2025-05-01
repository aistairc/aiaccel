import datetime

import toml

path = "pyproject.toml"
data = toml.load(path)
current_version = data["project"]["version"]

dt = datetime.datetime.now()
data["project"]["version"] = f"{dt.year}.{dt.month:02}"

with open(path, "w") as f:
    toml.dump(data, f)

print(f"Bumped version: {current_version} → {data['project']['version']}")
