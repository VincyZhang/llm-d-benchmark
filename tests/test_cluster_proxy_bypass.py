from __future__ import annotations

from kubernetes import client, config as k8s_config

from llmdbenchmark.utilities.cluster import _configure_proxy_bypass, kube_connect


def test_configure_proxy_bypass_adds_loopback_host_and_port(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "corp.example")
    monkeypatch.setenv("no_proxy", "shell.example")

    configuration = client.Configuration()
    configuration.host = "https://127.0.0.1:45645"
    configuration.proxy = "http://proxy.example:8080"

    _configure_proxy_bypass(configuration)

    assert configuration.proxy is None
    no_proxy_entries = set(configuration.no_proxy.split(","))
    assert {
        "corp.example",
        "shell.example",
        "localhost",
        "127.0.0.1",
        "::1",
        "localhost:45645",
        "127.0.0.1:45645",
        "::1:45645",
    }.issubset(no_proxy_entries)


def test_configure_proxy_bypass_ignores_non_loopback_hosts():
    configuration = client.Configuration()
    configuration.host = "https://10.0.0.1:6443"
    configuration.proxy = "http://proxy.example:8080"
    configuration.no_proxy = "existing.example"

    _configure_proxy_bypass(configuration)

    assert configuration.proxy == "http://proxy.example:8080"
    assert configuration.no_proxy == "existing.example"


def test_kube_connect_normalizes_default_configuration_for_loopback_kubeconfig(
    monkeypatch,
):
    def fake_load_kube_config(config_file=None, context=None):
        configuration = client.Configuration()
        configuration.host = "https://127.0.0.1:45645"
        configuration.proxy = "http://proxy.example:8080"
        client.Configuration.set_default(configuration)

    monkeypatch.setattr(k8s_config, "load_kube_config", fake_load_kube_config)
    monkeypatch.setattr(client, "ApiClient", lambda *args, **kwargs: object())

    kube_connect(kubeconfig="/tmp/fake-kubeconfig")

    default_configuration = client.Configuration.get_default_copy()
    assert default_configuration.proxy is None
    assert "127.0.0.1:45645" in (default_configuration.no_proxy or "")