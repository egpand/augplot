# Security Policy

## Supported versions

Augplot is currently in beta. Security fixes are provided for the latest `0.1.x`
release and the `main` branch.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |
| Earlier versions | No |

## Reporting a vulnerability

Once private vulnerability reporting is enabled, open the repository's
[Security page](https://github.com/egpand/augplot/security), select **Advisories**, and use
**Report a vulnerability**. Do not open a public issue for an undisclosed vulnerability.
If the private reporting option is unavailable, open an issue requesting a private contact
channel without including vulnerability details.

Include the affected version, impact, reproduction steps, and any suggested mitigation.
Remove API keys, credentials, personal data, and other secrets from reports and examples.
The maintainer aims to acknowledge reports within seven days and will coordinate fixes and
disclosure with the reporter.

## Security boundaries

Augplot sends bounded profiles of supplied data to the configured model provider. Reducing
or disabling sample rows does not anonymize all schema and summary information. Review the
provider's data-handling terms before using sensitive data.

Model-generated Python is validated before it runs locally, but this is defense in depth,
not an OS sandbox. Review [the generated-code guardrails](docs/generated-code-guardrails.md)
for the enforced restrictions and remaining limitations.
