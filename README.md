# smellai
master thesis


### How to run sonarqube
# 1. Kill and remove existing containers/images
docker stop sonarqube
docker rm sonarqube
docker rmi sonarqube

# 2. Clean up volumes (optional - removes all data)
docker volume prune

# 3. Fresh install with proper setup
docker run -d \
  --name sonarqube \
  -p 9000:9000 \
  -e SONAR_ES_BOOTSTRAP_CHECKS_DISABLE=true \
  sonarqube:latest

# 4. Wait for startup (check logs)
docker logs -f sonarqube
docker run --rm -v "$(pwd):/usr/src" --network="host" -e SONAR_HOST_URL="http://localhost:9000" -e SONAR_SCANNER_OPTS="-Dsonar.projectKey=sonar-test-app -Dsonar.java.binaries=. -Dsonar.language=java -Dsonar.verbose=true" -e SONAR_TOKEN="squ_5f53blabla" sonarsource/sonar-scanner-cli

# Wait a minute after "operational", then test
curl http://localhost:9000/api/system/status