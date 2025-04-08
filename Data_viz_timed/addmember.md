Pour ajouter un nouvel utilisateur à votre instance InfluxDB, vous rencontrez une erreur d'authentification (401 Unauthorized). Ce problème est généralement lié à des identifiants d'authentification manquants ou incorrects.

## Solution

Avant d'exécuter la commande pour créer un utilisateur, vous devez configurer l'authentification pour le CLI influx. Voici comment procéder :

1. **Configurez l'authentification pour le CLI influx** :

```sh
influx config create --config-name theoconfig \
  --host-url http://localhost:8086 \
  --org testinfluxdb2entreprise \
  --token `API_TOKEN` \
  --active
```

Remplacez :
- `theoconfig` : Nom de votre configuration
- `testinfluxdb2entreprise` : Nom de votre organisation
- `API_TOKEN` : Votre token API avec les permissions nécessaires

[Set up the influx CLI](https://docs.influxdata.com/influxdb/v2/tools/influx-cli/#set-up-the-influx-cli)

2. **activer la config si elle ne l'est pas**

```
influx config set --active --config-name theoconfig
```


3. **Après avoir configuré l'authentification**, réessayez votre commande de création d'utilisateur :

```
influx user create --org testinfluxdb2entreprise --name readuser --password readuserpwd
```

## Remarques importantes

- L'erreur 401 indique que vous n'êtes pas authentifié correctement pour effectuer cette opération.
- Assurez-vous que l'organisation "testinfluxdb2entreprise" existe bien. Si ce n'est pas le cas, vous devrez d'abord la créer.
- Vérifiez que votre token API a les permissions suffisantes pour créer des utilisateurs.
- Si vous utilisez InfluxDB v1, la méthode de création d'utilisateurs est différente et utilise des commandes InfluxQL comme `CREATE USER`.

Si vous avez besoin de créer une organisation, vous pouvez utiliser :

```
influx org create --name "testinfluxdb2entreprise"
```

[Create an organization using the influx CLI](https://docs.influxdata.com/influxdb/v2/admin/organizations/create-org/#create-an-organization-using-the-influx-cli)