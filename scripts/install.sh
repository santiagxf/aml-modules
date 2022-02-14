az login

RESOURCE_GROUP='<my-resource-group>'
WORKSPACE='<my-workspace-name>'
COMPONENTS_PATH='modules'
REPOSITORY_PATH='/home/santiagxf/repos/aml-modules/'

INSTALLED=$(az ml component list -g $RESOURCE_GROUP -w $WORKSPACE | jq .[].name)
COMP_FILES=$(find $COMPONENTS_PATH -exec realpath --relative-to=$REPOSITORY_PATH {} \;)

for COMP in $COMP_FILES
do
    COMPONENT_NAME=$(yq -r .name $COMP)
    COMPONENT_VERSION=$(yq -r .version $COMP)
    COMPONENT_URL=${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/blob/${GITHUB_REF##*/}/$COMP

    echo "::debug::Verifying component installation for $COMPONENT_NAME @ $COMPONENT_VERSION"
    COMPONENT_INSTALLED=$(echo $INSTALLED | jq --arg COMPONENT_NAME $COMPONENT_NAME 'select(contains($COMPONENT_NAME))')

    echo "::debug::Installing component $COMPONENT_NAME @ $COMPONENT_VERSION"
    if [ -n "$COMPONENT_INSTALLED" ];
    then
    echo "::debug::Installing and upgrading default version"
    az ml component create --file $COMP -g $RESOURCE_GROUP -w $WORKSPACE --label default
    else
    az ml component create --file $COMP -g $RESOURCE_GROUP -w $WORKSPACE
    fi
done