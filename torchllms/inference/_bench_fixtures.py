"""Large synthetic content for the throughput bench's W3 multiturn agentic
workload. Separated from ``throughput_bench.py`` to keep that file readable
— these are several-thousand-line string literals.

Three tool outputs, sized to be ~2000 tokens each in the o200k / cl100k
families. Content is plausible-looking but entirely synthetic — scenario
is a platform engineer debugging an intermittent 504 on an
``auth-service`` Kubernetes deployment.
"""

from __future__ import annotations

KUBECTL_AUTH_OUT = r"""$ kubectl top pods -n prod -l app=auth-service
NAME                               CPU(cores)   MEMORY(bytes)
auth-service-7d4b9c8f5-abc123      847m         1842Mi
auth-service-7d4b9c8f5-bc4d12      912m         1901Mi
auth-service-7d4b9c8f5-c5e8f9      124m         612Mi
auth-service-7d4b9c8f5-d7a2b1      118m         587Mi
auth-service-7d4b9c8f5-e9f3c4      132m         634Mi
auth-service-7d4b9c8f5-f1a4d5      109m         571Mi
auth-service-7d4b9c8f5-g2b5e6      127m         619Mi
auth-service-7d4b9c8f5-h3c6f7      141m         642Mi
auth-service-7d4b9c8f5-i4d7a8      115m         598Mi
auth-service-7d4b9c8f5-j5e8b9      122m         605Mi
auth-service-7d4b9c8f5-k6f9c0      138m         628Mi
auth-service-7d4b9c8f5-l7a0d1      130m         617Mi

$ kubectl describe pod auth-service-7d4b9c8f5-abc123 -n prod
Name:             auth-service-7d4b9c8f5-abc123
Namespace:        prod
Priority:         1000
Service Account:  auth-service-sa
Node:             ip-10-0-14-82.us-east-1.compute.internal/10.0.14.82
Start Time:       Wed, 22 Apr 2026 09:12:44 +0000
Labels:           app=auth-service
                  linkerd.io/control-plane-ns=linkerd
                  linkerd.io/proxy-deployment=auth-service
                  pod-template-hash=7d4b9c8f5
                  version=v2.4.1-rc3
Annotations:      linkerd.io/created-by: linkerd/proxy-injector stable-2.14.6
                  linkerd.io/inject: enabled
                  linkerd.io/proxy-version: stable-2.14.6
Status:           Running
IP:               10.0.18.112
IPs:
  IP:           10.0.18.112
Controlled By:  ReplicaSet/auth-service-7d4b9c8f5
Containers:
  auth-service:
    Container ID:   containerd://a8f29e1b3c0d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f
    Image:          registry.internal/auth-service:v2.4.1-rc3
    Image ID:       registry.internal/auth-service@sha256:3f8a9b2c1d0e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a
    Port:           8080/TCP
    Host Port:      0/TCP
    State:          Running
      Started:      Wed, 22 Apr 2026 09:12:58 +0000
    Ready:          True
    Restart Count:  2
    Limits:
      cpu:     2
      memory:  2Gi
    Requests:
      cpu:      500m
      memory:   1Gi
    Liveness:   http-get http://:8080/healthz delay=30s timeout=3s period=10s #success=1 #failure=3
    Readiness:  http-get http://:8080/readyz delay=5s timeout=2s period=5s #success=1 #failure=3
    Environment:
      DATABASE_URL:        <set to the key 'url' in secret 'auth-db-credentials'>
      JWT_SIGNING_KEY:     <set to the key 'key' in secret 'auth-jwt-keys'>
      REDIS_ADDR:          redis-auth.prod.svc.cluster.local:6379
      LOG_LEVEL:           info
      MAX_DB_CONNECTIONS:  25
      DB_IDLE_TIMEOUT:     30s
    Mounts:
      /etc/auth/tls from auth-tls-certs (ro)
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-xk7p2 (ro)
Conditions:
  Type                        Status
  PodReadyToStartContainers   True
  Initialized                 True
  Ready                       True
  ContainersReady             True
  PodScheduled                True
Events:
  Type     Reason     Age                   From     Message
  ----     ------     ----                  ----     -------
  Warning  Unhealthy  47m (x4 over 2h14m)   kubelet  Readiness probe failed: HTTP probe failed with statuscode: 503
  Warning  Unhealthy  23m (x2 over 41m)     kubelet  Liveness probe failed: Get "http://10.0.18.112:8080/healthz": context deadline exceeded
  Normal   Killing    23m                   kubelet  Container auth-service failed liveness probe, will be restarted
  Normal   Pulled     22m                   kubelet  Container image "registry.internal/auth-service:v2.4.1-rc3" already present on machine
  Normal   Created    22m                   kubelet  Created container auth-service
  Normal   Started    22m                   kubelet  Started container auth-service

$ kubectl logs auth-service-7d4b9c8f5-abc123 -n prod --tail=80 --since=10m
2026-04-22T14:18:02.142Z INFO  auth.handler POST /api/v1/login user_id=u_8821f91 latency_ms=43 status=200
2026-04-22T14:18:02.291Z INFO  auth.handler POST /api/v1/login user_id=u_42a91cd latency_ms=38 status=200
2026-04-22T14:18:03.004Z WARN  auth.db pool wait=812ms active=25 idle=0 waiting=14
2026-04-22T14:18:03.227Z INFO  auth.handler POST /api/v1/refresh user_id=u_5fb2c0e latency_ms=29 status=200
2026-04-22T14:18:04.118Z ERROR auth.db query failed: context deadline exceeded (pq: canceling statement due to conflict with recovery)
2026-04-22T14:18:04.119Z ERROR auth.handler POST /api/v1/login err="db query timeout" latency_ms=30017 status=504
2026-04-22T14:18:04.301Z INFO  auth.handler POST /api/v1/login user_id=u_91c3d72 latency_ms=52 status=200
2026-04-22T14:18:05.442Z WARN  auth.db pool wait=1204ms active=25 idle=0 waiting=18
2026-04-22T14:18:06.118Z ERROR auth.handler POST /api/v1/login err="upstream timeout" latency_ms=30002 status=504
  at github.com/internal/auth/db.(*Pool).Acquire(pool.go:142)
  at github.com/internal/auth/handler.LoginHandler(login.go:87)
  at net/http.HandlerFunc.ServeHTTP(server.go:2136)
  at github.com/internal/auth/middleware.Trace(trace.go:41)
2026-04-22T14:18:07.003Z INFO  auth.handler POST /api/v1/login user_id=u_2a8f4bd latency_ms=41 status=200
2026-04-22T14:18:07.812Z INFO  auth.handler POST /api/v1/validate user_id=u_7e1c3f9 latency_ms=12 status=200
2026-04-22T14:18:08.441Z WARN  auth.db slow_query duration_ms=4821 query="SELECT * FROM sessions WHERE token=$1"
2026-04-22T14:18:09.112Z ERROR auth.handler POST /api/v1/login err="db query timeout" latency_ms=30009 status=504
  at github.com/internal/auth/db.(*Pool).Query(pool.go:198)
  at github.com/internal/auth/handler.LoginHandler(login.go:94)
2026-04-22T14:18:10.227Z INFO  auth.handler POST /api/v1/login user_id=u_c4d2e81 latency_ms=47 status=200
2026-04-22T14:18:11.009Z INFO  auth.handler GET /api/v1/me user_id=u_a9b3f02 latency_ms=18 status=200
2026-04-22T14:18:12.338Z WARN  auth.db pool wait=1891ms active=25 idle=0 waiting=22
2026-04-22T14:18:13.102Z ERROR auth.handler POST /api/v1/login err="db query timeout" latency_ms=30014 status=504
2026-04-22T14:18:14.218Z INFO  auth.handler POST /api/v1/login user_id=u_14e8c20 latency_ms=39 status=200
2026-04-22T14:18:15.441Z INFO  auth.handler POST /api/v1/refresh user_id=u_67a1b94 latency_ms=33 status=200
2026-04-22T14:18:16.008Z WARN  linkerd-proxy outbound retry tap=auth-db-primary.prod:5432 reason="io error: connection reset by peer"
2026-04-22T14:18:17.112Z ERROR auth.handler POST /api/v1/login err="upstream timeout" latency_ms=30011 status=504
2026-04-22T14:18:18.301Z INFO  auth.handler POST /api/v1/login user_id=u_82f4e10 latency_ms=44 status=200
2026-04-22T14:18:19.004Z INFO  auth.handler POST /api/v1/validate user_id=u_33c9a21 latency_ms=14 status=200

$ kubectl get endpoints auth-service -n prod -o wide
NAME           ENDPOINTS                                                     AGE
auth-service   10.0.18.112:8080,10.0.18.134:8080,10.0.19.201:8080 + 9 more   84d

$ kubectl get svc auth-service -n prod -o yaml | head -60
apiVersion: v1
kind: Service
metadata:
  annotations:
    linkerd.io/inject: enabled
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
  creationTimestamp: "2026-01-28T17:04:12Z"
  labels:
    app: auth-service
    tier: platform
  name: auth-service
  namespace: prod
  resourceVersion: "88412901"
  uid: 3f9a2b1c-4d5e-6f7a-8b9c-0d1e2f3a4b5c
spec:
  clusterIP: 172.20.84.112
  clusterIPs:
  - 172.20.84.112
  internalTrafficPolicy: Cluster
  ipFamilies:
  - IPv4
  ipFamilyPolicy: SingleStack
  ports:
  - name: http
    port: 8080
    protocol: TCP
    targetPort: 8080
  - name: linkerd-admin
    port: 4191
    protocol: TCP
    targetPort: 4191
  selector:
    app: auth-service
  sessionAffinity: None
  type: ClusterIP
status:
  loadBalancer: {}
"""

BASH_NET_OUT = r"""$ ss -tnp | grep :8080 | head -40
ESTAB      0      0         10.0.18.112:40122     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=12))
ESTAB      0      0         10.0.18.112:40124     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=14))
ESTAB      0      0         10.0.18.112:40126     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=16))
ESTAB      0      0         10.0.18.112:40128     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=18))
ESTAB      0    128         10.0.18.112:40130     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=20))
ESTAB      0      0         10.0.18.112:40132     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=22))
CLOSE-WAIT 1      0         10.0.18.112:40134     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=24))
CLOSE-WAIT 1      0         10.0.18.112:40136     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=26))
CLOSE-WAIT 1      0         10.0.18.112:40138     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=28))
ESTAB      0      0         10.0.18.112:40140     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=30))
ESTAB      0   2144         10.0.18.112:40142     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=32))
TIME-WAIT  0      0         10.0.18.112:40144     10.0.4.18:8080
TIME-WAIT  0      0         10.0.18.112:40146     10.0.4.18:8080
TIME-WAIT  0      0         10.0.18.112:40148     10.0.4.22:8080
TIME-WAIT  0      0         10.0.18.112:40150     10.0.4.22:8080
ESTAB      0      0         10.0.18.112:40152     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=38))
ESTAB      0      0         10.0.18.112:40154     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=40))
ESTAB      0    512         10.0.18.112:40156     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=42))
ESTAB      0      0         10.0.18.112:40158     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=44))
CLOSE-WAIT 1      0         10.0.18.112:40160     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=46))
ESTAB      0      0         10.0.18.112:40162     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=48))
ESTAB      0      0         10.0.18.112:40164     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=50))
TIME-WAIT  0      0         10.0.18.112:40166     10.0.4.22:8080
TIME-WAIT  0      0         10.0.18.112:40168     10.0.4.18:8080
ESTAB      0      0         10.0.18.112:40170     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=56))
ESTAB      0   1024         10.0.18.112:40172     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=58))
ESTAB      0      0         10.0.18.112:40174     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=60))
CLOSE-WAIT 1      0         10.0.18.112:40176     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=62))
ESTAB      0      0         10.0.18.112:40178     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=64))
ESTAB      0      0         10.0.18.112:40180     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=66))
TIME-WAIT  0      0         10.0.18.112:40182     10.0.4.18:8080
TIME-WAIT  0      0         10.0.18.112:40184     10.0.4.22:8080
ESTAB      0      0         10.0.18.112:40186     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=72))
ESTAB      0    256         10.0.18.112:40188     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=74))
ESTAB      0      0         10.0.18.112:40190     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=76))
CLOSE-WAIT 1      0         10.0.18.112:40192     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=78))
ESTAB      0      0         10.0.18.112:40194     10.0.4.18:8080    users:(("auth-service",pid=1234,fd=80))
ESTAB      0      0         10.0.18.112:40196     10.0.4.22:8080    users:(("auth-service",pid=1234,fd=82))
TIME-WAIT  0      0         10.0.18.112:40198     10.0.4.18:8080
TIME-WAIT  0      0         10.0.18.112:40200     10.0.4.22:8080
... (2847 more)

$ netstat -s | grep -iE 'retransmit|drop|reset'
    4821 segments retransmitted
    128 resets received for embryonic SYN_RECV sockets
    TCPLostRetransmit: 412
    14821 connections reset due to unexpected data
    2104 connections reset due to early user close
    TCPAbortOnTimeout: 891
    TCPAbortOnClose: 2341
    TCPSynRetrans: 3218
    TCPFastRetrans: 1204
    18422 packets dropped from out-of-order queue due to socket buffer overrun
    TCPRcvQDrop: 441
    TCPBacklogDrop: 89

$ curl -s -w "@/etc/curl-format.txt" -o /dev/null https://auth-service.prod.svc.cluster.local:8080/api/v1/login
time_namelookup:  0.004218s
time_connect:     0.008914s
time_appconnect:  0.042108s
time_pretransfer: 0.042187s
time_starttransfer: 0.089412s
time_total:       0.091844s

$ curl -s -w "@/etc/curl-format.txt" -o /dev/null https://auth-service.prod.svc.cluster.local:8080/api/v1/login
time_namelookup:  0.003842s
time_connect:     0.008217s
time_appconnect:  0.041518s
time_pretransfer: 0.041601s
time_starttransfer: 28.412844s
time_total:       28.418217s

$ curl -s -w "@/etc/curl-format.txt" -o /dev/null https://auth-service.prod.svc.cluster.local:8080/api/v1/login
time_namelookup:  0.003914s
time_connect:     0.008412s
time_appconnect:  0.042207s
time_pretransfer: 0.042291s
time_starttransfer: 0.091218s
time_total:       0.093841s

$ curl -s -w "@/etc/curl-format.txt" -o /dev/null https://auth-service.prod.svc.cluster.local:8080/api/v1/login
time_namelookup:  0.004018s
time_connect:     0.008814s
time_appconnect:  0.042814s
time_pretransfer: 0.042891s
time_starttransfer: 30.002184s
time_total:       30.008441s

$ curl -s -w "@/etc/curl-format.txt" -o /dev/null https://auth-service.prod.svc.cluster.local:8080/api/v1/login
time_namelookup:  0.003918s
time_connect:     0.008614s
time_appconnect:  0.042017s
time_pretransfer: 0.042104s
time_starttransfer: 0.088412s
time_total:       0.090218s

$ dig +trace auth-service.prod.svc.cluster.local

; <<>> DiG 9.18.24 <<>> +trace auth-service.prod.svc.cluster.local
;; global options: +cmd
.                       518400  IN      NS      a.root-servers.net.
.                       518400  IN      NS      b.root-servers.net.
.                       518400  IN      NS      c.root-servers.net.
;; Received 239 bytes from 10.0.0.10#53(10.0.0.10) in 2 ms

cluster.local.          86400   IN      NS      ns1.cluster.local.
;; Received 89 bytes from 10.0.0.10#53(10.0.0.10) in 3 ms

svc.cluster.local.      30      IN      NS      kube-dns.kube-system.svc.cluster.local.
;; Received 112 bytes from 10.0.0.10#53(10.0.0.10) in 4 ms

auth-service.prod.svc.cluster.local. 30 IN A    172.20.84.112
;; Received 82 bytes from 10.0.0.10#53(kube-dns) in 5 ms

$ tcpdump -ni eth0 'port 8080' -c 30
tcpdump: verbose output suppressed, use -v[v]... for full protocol decode
listening on eth0, link-type EN10MB (Ethernet), snapshot length 262144 bytes
14:22:04.118241 IP 10.0.18.112.40422 > 10.0.4.18.8080: Flags [S], seq 2841029184, win 62727, options [mss 8961,sackOK,TS val 3891241802 ecr 0,nop,wscale 7], length 0
14:22:04.118918 IP 10.0.4.18.8080 > 10.0.18.112.40422: Flags [S.], seq 1841029841, ack 2841029185, win 62643, options [mss 8961,sackOK,TS val 1284018412 ecr 3891241802,nop,wscale 7], length 0
14:22:04.118942 IP 10.0.18.112.40422 > 10.0.4.18.8080: Flags [.], ack 1, win 491, options [nop,nop,TS val 3891241803 ecr 1284018412], length 0
14:22:04.119218 IP 10.0.18.112.40422 > 10.0.4.18.8080: Flags [P.], seq 1:512, ack 1, win 491, options [nop,nop,TS val 3891241803 ecr 1284018412], length 511
14:22:04.119841 IP 10.0.4.18.8080 > 10.0.18.112.40422: Flags [.], ack 512, win 489, options [nop,nop,TS val 1284018413 ecr 3891241803], length 0
14:22:04.162184 IP 10.0.4.18.8080 > 10.0.18.112.40422: Flags [P.], seq 1:841, ack 512, win 489, options [nop,nop,TS val 1284018456 ecr 3891241803], length 840
14:22:04.162214 IP 10.0.18.112.40422 > 10.0.4.18.8080: Flags [.], ack 841, win 498, options [nop,nop,TS val 3891241847 ecr 1284018456], length 0
14:22:04.162891 IP 10.0.18.112.40422 > 10.0.4.18.8080: Flags [F.], seq 512, ack 841, win 498, options [nop,nop,TS val 3891241847 ecr 1284018456], length 0
14:22:04.163418 IP 10.0.4.18.8080 > 10.0.18.112.40422: Flags [F.], seq 841, ack 513, win 489, options [nop,nop,TS val 1284018457 ecr 3891241847], length 0
14:22:04.163441 IP 10.0.18.112.40422 > 10.0.4.18.8080: Flags [.], ack 842, win 498, options [nop,nop,TS val 3891241848 ecr 1284018457], length 0
14:22:05.241841 IP 10.0.18.112.40424 > 10.0.4.22.8080: Flags [S], seq 3021848412, win 62727, options [mss 8961,sackOK,TS val 3891242926 ecr 0,nop,wscale 7], length 0
14:22:05.242814 IP 10.0.4.22.8080 > 10.0.18.112.40424: Flags [S.], seq 2018418941, ack 3021848413, win 62643, options [mss 8961,sackOK,TS val 2014218941 ecr 3891242926,nop,wscale 7], length 0
14:22:05.242841 IP 10.0.18.112.40424 > 10.0.4.22.8080: Flags [.], ack 1, win 491, options [nop,nop,TS val 2014218941], length 0
14:22:05.243218 IP 10.0.18.112.40424 > 10.0.4.22.8080: Flags [P.], seq 1:612, ack 1, win 491, options [nop,nop,TS val 3891242928 ecr 2014218941], length 611
14:22:05.243891 IP 10.0.4.22.8080 > 10.0.18.112.40424: Flags [.], ack 612, win 489, options [nop,nop,TS val 2014218942 ecr 3891242928], length 0
14:22:35.244118 IP 10.0.18.112.40424 > 10.0.4.22.8080: Flags [R.], seq 612, ack 1, win 491, options [nop,nop,TS val 3891272929 ecr 2014218942], length 0
14:22:36.118241 IP 10.0.18.112.40426 > 10.0.4.18.8080: Flags [S], seq 4218412091, win 62727, options [mss 8961,sackOK,TS val 3891273803 ecr 0,nop,wscale 7], length 0
14:22:36.118918 IP 10.0.4.18.8080 > 10.0.18.112.40426: Flags [S.], seq 3418929184, ack 4218412092, win 62643, options [mss 8961,sackOK,TS val 1284050412 ecr 3891273803,nop,wscale 7], length 0
14:22:36.118942 IP 10.0.18.112.40426 > 10.0.4.18.8080: Flags [.], ack 1, win 491, options [nop,nop,TS val 3891273804 ecr 1284050412], length 0
14:22:36.119218 IP 10.0.18.112.40426 > 10.0.4.18.8080: Flags [P.], seq 1:498, ack 1, win 491, options [nop,nop,TS val 3891273804 ecr 1284050412], length 497
14:22:36.119841 IP 10.0.4.18.8080 > 10.0.18.112.40426: Flags [.], ack 498, win 489, options [nop,nop,TS val 1284050413 ecr 3891273804], length 0
14:22:36.162184 IP 10.0.4.18.8080 > 10.0.18.112.40426: Flags [P.], seq 1:421, ack 498, win 489, options [nop,nop,TS val 1284050456 ecr 3891273804], length 420
14:22:36.162214 IP 10.0.18.112.40426 > 10.0.4.18.8080: Flags [.], ack 421, win 498, options [nop,nop,TS val 3891273848 ecr 1284050456], length 0
14:22:38.441218 IP 10.0.18.112.40428 > 10.0.4.22.8080: Flags [S], seq 1928412841, win 62727, options [mss 8961,sackOK,TS val 3891276127 ecr 0,nop,wscale 7], length 0
14:22:38.441891 IP 10.0.4.22.8080 > 10.0.18.112.40428: Flags [S.], seq 4091284184, ack 1928412842, win 62643, options [mss 8961,sackOK,TS val 2014252141 ecr 3891276127,nop,wscale 7], length 0
14:22:38.441914 IP 10.0.18.112.40428 > 10.0.4.22.8080: Flags [.], ack 1, win 491, options [nop,nop,TS val 3891276127 ecr 2014252141], length 0
14:22:38.442218 IP 10.0.18.112.40428 > 10.0.4.22.8080: Flags [P.], seq 1:584, ack 1, win 491, options [nop,nop,TS val 3891276128 ecr 2014252141], length 583
14:23:08.442841 IP 10.0.18.112.40428 > 10.0.4.22.8080: Flags [R.], seq 584, ack 1, win 491, options [nop,nop,TS val 3891306128 ecr 2014252141], length 0
30 packets captured
42 packets received by filter
0 packets dropped by kernel

$ grep -c "504" /var/log/nginx/access.log
8421

$ grep -c "upstream timed out" /var/log/nginx/error.log
2104

$ wc -l /var/log/nginx/access.log
1284918 /var/log/nginx/access.log
"""

PGSTAT_AUTH_OUT = r"""psql (15.5)
Type "help" for help.

auth=> SELECT pid, state, query_start, query FROM pg_stat_activity WHERE datname='auth' ORDER BY query_start LIMIT 20;
  pid  |        state        |         query_start          |                                              query
-------+---------------------+------------------------------+--------------------------------------------------------------------------------------------------
 18234 | idle in transaction | 2026-04-22 14:31:02.448192+00 | SELECT id, user_id, token_hash, expires_at FROM auth_sessions WHERE user_id = $1 FOR UPDATE
 18241 | idle in transaction | 2026-04-22 14:31:04.771338+00 | UPDATE auth_sessions SET last_seen_at = now() WHERE id = $1
 18256 | idle in transaction | 2026-04-22 14:31:07.902114+00 | SELECT id FROM auth_sessions WHERE token_hash = $1 AND expires_at > now()
 18262 | active              | 2026-04-22 14:31:09.118847+00 | SELECT s.id, s.user_id, u.email, u.mfa_enabled FROM auth_sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = $1
 18277 | idle in transaction | 2026-04-22 14:31:11.334729+00 | INSERT INTO auth_audit_log (user_id, event, ip, created_at) VALUES ($1,$2,$3, now())
 18283 | active              | 2026-04-22 14:31:12.550021+00 | DELETE FROM auth_sessions WHERE expires_at < now() - interval '7 days'
 18291 | idle in transaction | 2026-04-22 14:31:13.880412+00 | SELECT count(*) FROM auth_sessions WHERE user_id = $1
 18299 | idle               | 2026-04-22 14:31:14.664180+00 | COMMIT
 18304 | active              | 2026-04-22 14:31:15.221903+00 | SELECT id, user_id, token_hash, expires_at, refresh_count FROM auth_sessions WHERE token_hash = $1
 18311 | idle in transaction | 2026-04-22 14:31:16.097554+00 | SELECT id FROM auth_sessions WHERE user_id = $1 AND device_id = $2 FOR UPDATE
 18318 | active              | 2026-04-22 14:31:17.412876+00 | UPDATE users SET last_login_at = now(), login_count = login_count + 1 WHERE id = $1
 18324 | idle in transaction | 2026-04-22 14:31:18.773990+00 | SELECT 1 FROM auth_sessions WHERE user_id = $1 LIMIT 1
 18331 | active              | 2026-04-22 14:31:19.881115+00 | SELECT pg_advisory_xact_lock($1)
 18338 | idle in transaction | 2026-04-22 14:31:20.445209+00 | SELECT token_hash FROM auth_sessions WHERE id = $1 FOR NO KEY UPDATE
 18345 | idle               | 2026-04-22 14:31:21.009673+00 | BEGIN
 18351 | active              | 2026-04-22 14:31:22.118441+00 | SELECT id, user_id, created_at FROM auth_sessions WHERE ip_address = $1 ORDER BY created_at DESC LIMIT 50
 18358 | idle in transaction | 2026-04-22 14:31:23.662907+00 | UPDATE auth_sessions SET refresh_count = refresh_count + 1 WHERE id = $1
 18362 | active              | 2026-04-22 14:31:24.771802+00 | SELECT COUNT(*) FROM auth_sessions s WHERE s.user_id = $1 AND s.expires_at > now()
 18369 | idle in transaction | 2026-04-22 14:31:25.998123+00 | SELECT id FROM users WHERE email = lower($1)
 18374 | active              | 2026-04-22 14:31:26.445009+00 | SELECT * FROM auth_sessions WHERE token_hash = $1
(20 rows)

auth=> SELECT query, calls, total_time, mean_time, max_time FROM pg_stat_statements WHERE query LIKE '%auth_sessions%' ORDER BY mean_time DESC LIMIT 10;
                                             query                                              |  calls   |  total_time  | mean_time | max_time
-------------------------------------------------------------------------------------------------+----------+--------------+-----------+----------
 SELECT s.id, s.user_id, u.email FROM auth_sessions s JOIN users u ON u.id=s.user_id WHERE s.tok | 18422    | 41238712.31  | 2239.118  | 3487.442
 DELETE FROM auth_sessions WHERE expires_at < now() - interval $1                                | 982      | 1872344.008  | 1906.664  | 3102.881
 SELECT id, user_id, token_hash, expires_at FROM auth_sessions WHERE user_id = $1 FOR UPDATE     | 127331   | 217655893.12 | 1709.442  | 2988.119
 UPDATE auth_sessions SET refresh_count = refresh_count + $1 WHERE id = $2                       | 89217    | 118822713.45 | 1331.907  | 2644.018
 SELECT count(*) FROM auth_sessions WHERE user_id = $1 AND expires_at > now()                    | 201113   | 226773301.88 | 1127.550  | 2381.117
 SELECT id FROM auth_sessions WHERE user_id=$1 AND device_id=$2 FOR UPDATE                       | 42118    | 39188772.30  | 930.331   | 2101.778
 SELECT token_hash FROM auth_sessions WHERE id = $1 FOR NO KEY UPDATE                            | 55620    | 40098119.02  | 721.118   | 1845.662
 INSERT INTO auth_sessions (user_id, token_hash, expires_at) VALUES ($1,$2,$3) RETURNING id      | 38291    | 18722334.91  | 488.885   | 1632.019
 SELECT 1 FROM auth_sessions WHERE user_id = $1 LIMIT 1                                          | 812342   | 289733311.23 | 356.771   | 1248.992
 UPDATE auth_sessions SET last_seen_at = now() WHERE id = $1                                     | 901225   | 221998132.44 | 246.332   | 988.114
(10 rows)

$ tail -n 40 /var/log/postgresql/postgres-15-main.log
2026-04-22 14:29:58.118 UTC [18114] LOG:  duration: 2881.449 ms  statement: SELECT s.id, s.user_id, u.email, u.mfa_enabled FROM auth_sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = $1
2026-04-22 14:30:02.331 UTC [18128] LOG:  duration: 1523.778 ms  statement: SELECT id, user_id, token_hash, expires_at FROM auth_sessions WHERE user_id = $1 FOR UPDATE
2026-04-22 14:30:05.662 UTC [18133] LOG:  duration: 3487.442 ms  statement: SELECT s.id, s.user_id, u.email FROM auth_sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash = $1
2026-04-22 14:30:07.009 UTC [18141] LOG:  duration: 18.112 ms  statement: SELECT 1 FROM auth_sessions WHERE user_id = $1 LIMIT 1
2026-04-22 14:30:08.441 UTC [18155] LOG:  duration: 1902.119 ms  statement: DELETE FROM auth_sessions WHERE expires_at < now() - interval '7 days'
2026-04-22 14:30:10.882 UTC [18162] WARNING:  there is already a transaction in progress
2026-04-22 14:30:12.117 UTC [18168] LOG:  duration: 2119.773 ms  statement: UPDATE auth_sessions SET refresh_count = refresh_count + 1 WHERE id = $1
2026-04-22 14:30:14.445 UTC [18174] LOG:  duration: 1234.567 ms  statement: SELECT * FROM auth_sessions WHERE user_id = $1
2026-04-22 14:30:15.772 UTC [18182] ERROR:  deadlock detected
2026-04-22 14:30:15.772 UTC [18182] DETAIL:  Process 18182 waits for ShareLock on transaction 49218773; blocked by process 18133.
	Process 18133 waits for ShareLock on transaction 49218802; blocked by process 18182.
	Process 18182: UPDATE auth_sessions SET refresh_count = refresh_count + 1 WHERE id = $1
	Process 18133: UPDATE auth_sessions SET last_seen_at = now() WHERE id = $1
2026-04-22 14:30:15.772 UTC [18182] HINT:  See server log for query details.
2026-04-22 14:30:15.772 UTC [18182] CONTEXT:  while updating tuple (1082,17) in relation "auth_sessions"
2026-04-22 14:30:15.772 UTC [18182] STATEMENT:  UPDATE auth_sessions SET refresh_count = refresh_count + 1 WHERE id = $1
2026-04-22 14:30:17.221 UTC [18191] LOG:  duration: 887.441 ms  statement: SELECT count(*) FROM auth_sessions WHERE user_id = $1 AND expires_at > now()
2026-04-22 14:30:19.118 UTC [18198] LOG:  duration: 15.332 ms  statement: COMMIT
2026-04-22 14:30:21.445 UTC [18204] LOG:  duration: 2441.009 ms  statement: SELECT id, user_id, token_hash, expires_at FROM auth_sessions WHERE user_id = $1 FOR UPDATE
2026-04-22 14:30:23.662 UTC [18212] LOG:  unexpected EOF on client connection with an open transaction
2026-04-22 14:30:23.662 UTC [18212] LOG:  could not receive data from client: Connection reset by peer
2026-04-22 14:30:25.117 UTC [18221] LOG:  duration: 1677.881 ms  statement: SELECT s.id, s.user_id, u.email, u.mfa_enabled FROM auth_sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = $1
2026-04-22 14:30:27.334 UTC [18228] LOG:  duration: 341.229 ms  statement: UPDATE auth_sessions SET last_seen_at = now() WHERE id = $1
2026-04-22 14:30:29.662 UTC [18234] LOG:  duration: 2993.118 ms  statement: SELECT s.id, s.user_id FROM auth_sessions s WHERE s.token_hash = $1
2026-04-22 14:30:31.009 UTC [18241] LOG:  duration: 218.441 ms  statement: INSERT INTO auth_audit_log (user_id, event, ip, created_at) VALUES ($1,$2,$3,now())
2026-04-22 14:30:33.118 UTC [18247] LOG:  duration: 1449.772 ms  statement: SELECT token_hash FROM auth_sessions WHERE id = $1 FOR NO KEY UPDATE
2026-04-22 14:30:35.445 UTC [18255] LOG:  duration: 3287.661 ms  statement: SELECT s.id, s.user_id, u.email FROM auth_sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash = $1
2026-04-22 14:30:37.772 UTC [18262] LOG:  duration: 88.114 ms  statement: SELECT id FROM users WHERE email = lower($1)
2026-04-22 14:30:39.118 UTC [18268] LOG:  duration: 1918.775 ms  statement: SELECT count(*) FROM auth_sessions WHERE user_id = $1
2026-04-22 14:30:41.445 UTC [18274] LOG:  unexpected EOF on client connection with an open transaction
2026-04-22 14:30:43.662 UTC [18281] LOG:  duration: 2212.118 ms  statement: UPDATE auth_sessions SET refresh_count = refresh_count + 1 WHERE id = $1
2026-04-22 14:30:45.009 UTC [18288] LOG:  duration: 24.661 ms  statement: BEGIN
2026-04-22 14:30:47.118 UTC [18294] LOG:  duration: 1132.441 ms  statement: SELECT * FROM auth_sessions WHERE token_hash = $1
2026-04-22 14:30:49.445 UTC [18301] LOG:  duration: 3102.881 ms  statement: DELETE FROM auth_sessions WHERE expires_at < now() - interval '7 days'
2026-04-22 14:30:51.662 UTC [18309] LOG:  duration: 441.118 ms  statement: UPDATE users SET last_login_at = now() WHERE id = $1
2026-04-22 14:30:53.118 UTC [18316] LOG:  duration: 1771.229 ms  statement: SELECT id, user_id, token_hash FROM auth_sessions WHERE user_id = $1 FOR UPDATE
2026-04-22 14:30:55.334 UTC [18322] LOG:  duration: 16.009 ms  statement: SELECT 1 FROM auth_sessions WHERE user_id = $1 LIMIT 1
2026-04-22 14:30:57.662 UTC [18329] LOG:  duration: 2488.771 ms  statement: SELECT s.id, u.email FROM auth_sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash = $1
2026-04-22 14:30:59.118 UTC [18335] LOG:  duration: 778.441 ms  statement: INSERT INTO auth_sessions (user_id, token_hash, expires_at) VALUES ($1,$2,$3) RETURNING id
2026-04-22 14:31:01.445 UTC [18342] LOG:  duration: 1509.118 ms  statement: SELECT id, user_id, token_hash, expires_at FROM auth_sessions WHERE user_id = $1 FOR UPDATE
2026-04-22 14:31:03.662 UTC [18349] LOG:  duration: 2881.229 ms  statement: SELECT s.id, s.user_id, u.email FROM auth_sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash = $1
2026-04-22 14:31:05.118 UTC [18356] LOG:  duration: 341.009 ms  statement: UPDATE auth_sessions SET last_seen_at = now() WHERE id = $1

$ tail -n 25 /var/log/auth-service/app.log
2026-04-22T14:30:41.118Z INFO  AuthenticateHandler rid=a3f8e2c1 user=usr_82119 ip=10.0.14.211 action=login.start
2026-04-22T14:30:41.229Z INFO  AuthenticateHandler rid=a3f8e2c1 pool=acquire waiting=3 active=18 pool_size=20
2026-04-22T14:30:41.882Z INFO  AuthenticateHandler rid=a3f8e2c1 db.query=select_session duration_ms=612
2026-04-22T14:30:42.118Z INFO  AuthenticateHandler rid=a3f8e2c1 user=usr_82119 action=login.success jwt_issued=1
2026-04-22T14:30:42.441Z INFO  AuthenticateHandler rid=b7c2e118 user=usr_44128 ip=10.0.14.219 action=refresh.start
2026-04-22T14:30:43.009Z WARN  db.pool rid=b7c2e118 pool exhausted (pool_size=20, active=20, waiting=12)
2026-04-22T14:30:43.117Z WARN  db.pool rid=c1f9a204 pool exhausted (pool_size=20, active=20, waiting=13)
2026-04-22T14:30:43.662Z INFO  AuthenticateHandler rid=d2e4b771 user=usr_91182 ip=10.0.15.33 action=login.start
2026-04-22T14:30:44.118Z WARN  db.pool rid=d2e4b771 pool exhausted (pool_size=20, active=20, waiting=14)
2026-04-22T14:30:44.445Z ERROR AuthenticateHandler rid=b7c2e118 deadline exceeded while acquiring DB connection after 3000ms
  Traceback (most recent call last):
    File "/app/auth_service/handlers.py", line 128, in authenticate
      async with self.db.acquire(timeout=3.0) as conn:
    File "/app/auth_service/db/pool.py", line 84, in acquire
      raise ConnectionTimeoutError("deadline exceeded")
  auth_service.db.pool.ConnectionTimeoutError: deadline exceeded
2026-04-22T14:30:44.771Z INFO  AuthenticateHandler rid=e8a1c339 user=usr_33012 ip=10.0.15.88 action=login.success jwt_issued=1
2026-04-22T14:30:45.009Z WARN  db.pool rid=f4d2a108 pool exhausted (pool_size=20, active=20, waiting=15)
2026-04-22T14:30:45.441Z ERROR AuthenticateHandler rid=d2e4b771 deadline exceeded while acquiring DB connection after 3000ms
  Traceback (most recent call last):
    File "/app/auth_service/handlers.py", line 128, in authenticate
      async with self.db.acquire(timeout=3.0) as conn:
    File "/app/auth_service/db/pool.py", line 84, in acquire
      raise ConnectionTimeoutError("deadline exceeded")
  auth_service.db.pool.ConnectionTimeoutError: deadline exceeded
2026-04-22T14:30:46.118Z INFO  AuthenticateHandler rid=0a1b2c3d user=usr_77412 ip=10.0.15.102 action=refresh.success
2026-04-22T14:30:46.662Z WARN  db.pool rid=9f8e7d6c pool exhausted (pool_size=20, active=20, waiting=11)
2026-04-22T14:30:47.009Z ERROR AuthenticateHandler rid=f4d2a108 deadline exceeded while acquiring DB connection after 3000ms
2026-04-22T14:30:47.441Z INFO  AuthenticateHandler rid=5c4b3a29 user=usr_10221 ip=10.0.16.14 action=logout.success
2026-04-22T14:30:48.118Z WARN  linkerd-proxy upstream=auth-db.svc.cluster.local:5432 response 504 after 3012ms
2026-04-22T14:30:48.662Z INFO  AuthenticateHandler rid=ee11ff22 user=usr_56778 ip=10.0.16.81 action=login.success jwt_issued=1

auth=> SELECT * FROM pg_stat_database WHERE datname='auth';
 datid |  datname  | numbackends | xact_commit | xact_rollback | blks_read  | blks_hit   | tup_returned | tup_fetched | tup_inserted | tup_updated | tup_deleted | conflicts | temp_files | temp_bytes | deadlocks | stats_reset
-------+-----------+-------------+-------------+---------------+------------+------------+--------------+-------------+--------------+-------------+-------------+-----------+------------+------------+-----------+-----------------------------
 16485 | auth      |          87 |   128772334 |         42119 | 2881772344 | 9987712234 |  18827731044 |   882733119 |     44812229 |    82441097 |     7712034 |         0 |       1184 |  2881774112 |        44 | 2026-04-15 02:14:18.118+00
(1 row)

$ grep -c 'slow query' /var/log/postgresql/postgres-15-main.log
1872
"""


# Tool outputs in indexed form so the bench fixture can pull them by index.
TOOL_OUTPUTS = [KUBECTL_AUTH_OUT, PGSTAT_AUTH_OUT, BASH_NET_OUT]
