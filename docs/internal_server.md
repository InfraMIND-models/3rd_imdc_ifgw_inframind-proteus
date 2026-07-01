# Internal server for large data storage

> This file is for reference by the team members. Access to the server is not required for reproducing results.

We keep our large data files in an internal server called "La berenjena", 
managed and versioned via [DVC](https://dvc.org/). This server has no public access.


To sync data with the server, you need to create an ssh host alias named "la-berenjena".
On `~/.ssh/config`, add this (replacing with your credentials):
```
Host la-berenjena
    HostName [La Berenjena IP address]
    User [your username]
```

Once configured, you can get all large data files managed via DVC with:
```
dvc pull
```

