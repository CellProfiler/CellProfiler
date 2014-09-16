<?xml version="1.0"?>
<!--
This file is applied to xml test results in target/test-results/ to condense
As specified in the check-tests target in build.xml.
-->
<xsl:stylesheet version="1.0"
 xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/testsuite">
   <testsuite>
    <xsl:apply-templates select="testcase"/>
   </testsuite>
  </xsl:template>
  
  <xsl:template match="testcase">
   <xsl:apply-templates select="error"/>
   <xsl:apply-templates select="failure"/>
  </xsl:template>

  <xsl:template match="error">
   <testcase>
    <failure>
     <xsl:value-of select="parent::*/attribute::classname"/>.
     <xsl:value-of select="parent::*/attribute::name"/>
    </failure>
   </testcase>
  </xsl:template>

  <xsl:template match="failure">
   <testcase>
    <failure>
     <xsl:value-of select="parent::*/attribute::classname"/>.
     <xsl:value-of select="parent::*/attribute::name"/>
    </failure>
   </testcase>
  </xsl:template>

</xsl:stylesheet>