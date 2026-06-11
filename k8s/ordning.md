# Persitent volym och certifikat
```
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/certificate.yaml
```
#
# Alternativ 1 
Om du ska bygga vektordatabasen på klustret
```
kubectl apply -f k8s/chroma-job.yaml
```
#
# Alternativ 2
Kopiering av färdig data
```
kubectl apply -f pvc-loader.yaml
kubectl cp ./data/. pvc-loader:/data -n pts-2026-rictjo
kubectl cp ./chroma_db/. pvc-loader:/chroma_db -n pts-2026-rictjo
kubectl get pod pvc-loader -n pts-2026-rictjo
kubectl delete pod pvc-loader -n pts-2026-rictjo
```
# PVC lösningen
```
PVC (disk)
 ├── [kopierat via loader]
 │     ├── data/
 │     └── chroma_db/
 │
 └── mountas i app som:
       /app/data
       /app/chroma_db
```
# Deploy av lösningen
```
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```
