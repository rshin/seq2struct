#!/bin/bash

source argparser.sh

echo 'Test case: testFunc -h --params --longarg --longargvalue=someVal --longargspace="space & equal=xyz" --longvaluefollow "longvalue" simplearg "composite arg" -a -bc -D'
echo
echo 'parse_args "$@"'
echo

function testFunc() {
	parse_args "$@"

	echo "\$longarg = $longarg"
	echo "\$longargvalue = $longargvalue"
	echo "\$longargspace = $longargspace"
	echo "\$longvaluefollow = $longvaluefollow"
	echo
	echo "\$opta = $opta"
	echo "\$optb = $optb"
	echo "\$optc = $optc"
	echo "\$optd = $optd"
	echo "\$optD = $optD"
	echo "\$optE = $optE"
	echo
	echo "\$argument1 = $argument1"
	echo "\$argument2 = $argument2"
	echo
	echo "Test mapping -h to --help"
	echo "\$opth = $opth"
	echo "\$help = $help"
	echo
	echo "Test mapping --params to -p"
	echo "\$optp = $optp"
	echo "\$params = $params"
	echo

	if $opta; then
		echo "-a activated"
	fi

	#if $optb == true then
	$optb && echo "-b activated"
	! $optx && echo "-x not activated"

	#if $optc == true then
	[ $optc ] && echo "-c activated"
	[ ! $optc ] && echo "-y activated"

	#if $optb && $optc then
	$optb && $optc && echo "-bc activated"

	#if ! $optd then
	$optd || echo "-d not activated"

	if [ "$optD" == true ]; then
		echo "-D activated"
	fi
}

declare -A ARGPARSER_MAP
ARGPARSER_MAP=(
	[h]=help
	[p]=params
)

testFunc -h --params --longarg --longargvalue=someVal --longargspace="space & equal=xyz" --longvaluefollow "longvalue" simplearg "composite arg" -a -bc -D
